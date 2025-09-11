import copy
import json
import os
from collections import OrderedDict
from datasets import Dataset, DatasetDict

import math
import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir, tokenizer):

        self.client_id = client_id
        self.model = model          
        self.data_path = './data_train/8'

        self.local_train_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.local_train_data = load_dataset("json", data_files=self.local_train_data_path)

        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))
        self.data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )

        self.baseline_train_data_path = os.path.join("./data_train/initialized_data/8/remain1/8", "local_training_{}.json".format(self.client_id))


        with open(self.baseline_train_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) == 0:
            self.baseline_train_data = DatasetDict({"train": Dataset.from_dict({"dummy_column": []})})
        else:
            self.baseline_train_data = load_dataset("json", data_files=self.baseline_train_data_path)

    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):

        if local_val_set_size > 0:
            local_train_val = self.local_train_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_train_data["train"].shuffle().map(generate_and_tokenize_prompt)
            self.local_eval_dataset = None

            if len(self.baseline_train_data["train"]) > 0:
                self.baseline_train_dataset = (
                    self.baseline_train_data["train"]
                    .shuffle()
                    .map(generate_and_tokenize_prompt)
                )
            else:
                self.baseline_train_dataset = Dataset.from_dict({"dummy_column": []})

        self.local_val_set_size = local_val_set_size

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=375,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,

            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )

        self.local_trainer = transformers.Trainer(model=self.model,
                                                  train_dataset=self.local_train_dataset,
                                                  eval_dataset=self.local_eval_dataset,
                                                  args=self.train_args,
                                                  data_collator=transformers.DataCollatorForSeq2Seq(
                                                      tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                                                  ),
                                                  )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                        "default" in name))
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

    def train(self):
        self.local_trainer.train()


    def dynamic_train(self, epoch):
        self.local_trainer.train()
        local_perplexity_data_path = os.path.join(self.data_path, f"local_training_{self.client_id}_v1_{epoch}.json")
        self.compute_and_record_perplexities(local_perplexity_data_path, "local_perplexity", self.model)


    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")

        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id

    def terminate_training(self):
        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        return self.model

    def compute_sample_perplexity(self, sample: dict, model_to_use: torch.nn.Module) -> float:
        """
        Calculate the perplexity of a single sample (defined as exp(loss)).
        If the sample has not been tokenized (lacks the input_ids field),
        assume that the 'text' field exists in the sample and perform tokenization.
        """
        keys_to_keep = {"input_ids", "attention_mask", "labels"}
        sample = {k: v for k, v in sample.items() if k in keys_to_keep}

        batch = self.data_collator([sample])
        batch = {k: (v.to(model_to_use.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}

        model_to_use.eval()
        with torch.no_grad():
            outputs = model_to_use(**batch)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                raise ValueError("Model did not return loss.")
        perplexity = math.exp(loss.item())
        return perplexity

    def compute_and_record_perplexities(self, output_json_path: str, prop_name: str, model_to_use: torch.nn.Module):
        """
        Iterate through the local_train_dataset, calculate the perplexity for each sample,
        write this perplexity into each sample's record as a prop_name attribute,
        and finally write the updated data to output_json_path.
        """
        data = [sample for sample in self.baseline_train_dataset]
        if not data:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=4, ensure_ascii=False)
            print(f"The data is empty, and the empty file has been saved: {output_json_path}")
        else:
            for idx, sample in enumerate(data):
                try:
                    perplexity = self.compute_sample_perplexity(sample, model_to_use)
                except Exception as e:
                    print(f"Error computing perplexity for sample {idx}: {e}")
                    perplexity = None
                sample[prop_name] = perplexity

            dir_path = os.path.dirname(output_json_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"{prop_name} Record written {output_json_path}")
