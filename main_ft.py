import os
from typing import List
from tqdm import tqdm
import fire
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient
import datasets
from utils.prompter import Prompter
from peft import PeftModel
from peft import (
    set_peft_model_state_dict,
)


datasets.utils.logging.set_verbosity_error()

def fl_finetune(
        # model/data params
        global_model: str = './alpaca-7b-native',
        lora_model_path: str = './lora-our-method/8/19/',
        data_path: str = './data',
        output_dir: str = './lora-personalization/',
        local_val_data_root: str = './data',

        # FL hyperparams
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 1.0,
        num_communication_rounds: int = 20,
        num_clients: int = 8,

        # Local training hyperparams
        local_batch_size: int = 32,
        local_micro_batch_size: int = 4,
        local_num_epochs: int = 1,
        local_learning_rate: float = 2e-4,
        local_val_set_size: int = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,

        # LoRA hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj","v_proj"
        ],

        # llm hyperparams 大模型超参数
        train_on_inputs: bool = True,
        group_by_length: bool = True,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"

    # 确定数据路径是否存在
    data_path = os.path.join(data_path, str(num_clients))
    #assert (os.path.exists(data_path), "Please generate the data files for each client")
    assert os.path.exists(data_path), "Please generate the data files for each client"
    print(data_path)


    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = LlamaForCausalLM.from_pretrained(
        global_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        vocab_size=32001
    )

    tokenizer = LlamaTokenizer.from_pretrained(global_model)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    single_output_dir = os.path.join(lora_model_path, "adapter_model.bin")

    single_weights = torch.load(single_output_dir)
    set_peft_model_state_dict(model, single_weights, "default")


    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # 开始联邦微调过程
    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))
    print("output输出地址：",output_dir)

    num_local_epochs = 10
    for client_id in range(0, num_clients):
        print(f"\nInitializing Client_{client_id}")

        client = GeneralClient(client_id, model, data_path, output_dir, tokenizer)

        print(f"\nPreparing the local dataset and trainer for Client_{client_id}")
        client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
        client.build_local_trainer(tokenizer,
                                   local_micro_batch_size,
                                   gradient_accumulation_steps,
                                   1,
                                   local_learning_rate,
                                   group_by_length,
                                   ddp)

        print(f"Initiating the local training of Client_{client_id}")
        client.initiate_local_training()

        client_output_dir = os.path.join(output_dir, f"client_{client_id}_1")
        os.makedirs(client_output_dir, exist_ok=True)

        for epoch in range(num_local_epochs):
            client.train()

            local_val_data_path = os.path.join(local_val_data_root, str(num_clients), f"local_test_{client_id}.json")
            eval_loss = global_evaluation(model, local_val_data_path, generate_and_tokenize_prompt, 1, 'cuda')
            print(f"Client {client_id}, Epoch {epoch + 1}, Eval Loss: {eval_loss}")

            loss_log_path = os.path.join(client_output_dir, f"client_{client_id}_loss_log.txt")
            with open(loss_log_path, "a") as f:
                f.write(f"Epoch: {epoch}, Eval Loss: {eval_loss}\n")

            epoch_dir = os.path.join(client_output_dir, f"epoch_{epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)

            model_save_path = os.path.join(epoch_dir, "adapter_model.bin")
            torch.save(model.state_dict(), model_save_path)

            config_save_path = os.path.join(epoch_dir)
            config.save_pretrained(config_save_path) 

        print(f"\nTerminating the local training of Client_{client_id}")
        model = client.terminate_training()
        del client

    print("Local training completed for all clients!")


if __name__ == "__main__":
    fire.Fire(fl_finetune)
