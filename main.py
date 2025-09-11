import json
import os
from typing import List

import math
from tqdm import tqdm
import fire
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient, process_client_data_by_score
import datasets
from utils.prompter import Prompter
import sys
import shutil

sys.setrecursionlimit(5000)

datasets.utils.logging.set_verbosity_error()


def fl_finetune(
        # model/data parameters
        global_model: str = './alpaca-7b-native',
        data_path: str = './data_train/initialized_data/8/global1',
        output_dir: str = './FedMIDE/',
        val_data_root: str = './data',

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

        # llm hyperparams
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


    data_path = os.path.join(data_path, str(num_clients))
    assert os.path.exists(data_path), "Please generate the data files for each client"

    # Set up global_model and Tokenizer
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

    # Initialize the LlamaTokenizer
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
                                                                    ]  # could be sped up, probably
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
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))
    print("output_dir：",output_dir)

    for epoch in tqdm(range(num_communication_rounds)):
        print("Conducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)

        for client_id in selected_clients_set:
            client = GeneralClient(client_id, model, data_path, output_dir, tokenizer)
            print("Preparing the local dataset and trainer for Client_{}".format(client_id))

            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(tokenizer,
                                       local_micro_batch_size,
                                       gradient_accumulation_steps,
                                       local_num_epochs,
                                       local_learning_rate,
                                       group_by_length,
                                       ddp)


            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            if epoch > 0:
                dynamic_data_path = './data_train/8'
                old_local_perplexity_data_path = os.path.join(dynamic_data_path, f"local_training_{client_id}_v1_{epoch - 1}.json")
                aggregated_perplexity_data_path = os.path.join(dynamic_data_path, f"local_training_{client_id}_v2_{epoch - 1}.json")
                client.compute_and_record_perplexities(aggregated_perplexity_data_path, "aggregated_perplexity", model)
                personalization_score_path = os.path.join(dynamic_data_path, f"local_training_{client_id}_v3_{epoch - 1}.json")
                map_scores_to_json_from_two_json(old_local_perplexity_data_path, aggregated_perplexity_data_path, personalization_score_path)
                process_client_data_by_score(epoch, client_id)

            print("Local training starts ... ")
            client.dynamic_train(epoch)

            print("Terminating the local training of Client_{}\n".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            del client

        print("Collecting the weights of clients and performing aggregation")
        model = FedAvg(model,
                       selected_clients_set,
                       output_dir,
                       local_dataset_len_dict,
                       epoch,
                       )
        torch.save(model.state_dict(), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        config.save_pretrained(output_dir)

        val_data_path = os.path.join(val_data_root, str(num_clients), 'global_test.json')
        eval_loss = global_evaluation(model, val_data_path, generate_and_tokenize_prompt, 1, 'cuda')
        print('communication round: ', epoch, ' the eval loss: ', eval_loss)

        loss_path = os.path.join(output_dir, "eval_loss.txt")
        with open(loss_path, "a") as f:
            f.write(f"Global Epoch: {epoch}  Eval Loss: {eval_loss.item() if hasattr(eval_loss, 'item') else eval_loss}\n")

        data_dir = './data_train/8'

        if epoch > 0:
            for client_id in range(num_clients):
                file_v1_path = os.path.join(data_dir, f"local_training_{client_id}_v1_{epoch - 1}.json")
                file_v2_path = os.path.join(data_dir, f"local_training_{client_id}_v2_{epoch - 1}.json")
                file_v3_path = os.path.join(data_dir, f"local_training_{client_id}_v3_{epoch - 1}.json")
                if os.path.exists(file_v1_path):
                    os.remove(file_v1_path)
                    print(f"Deleted files: {file_v1_path}")
                else:
                    print(f"file does not exist: {file_v1_path}")

                if os.path.exists(file_v2_path):
                    os.remove(file_v2_path)
                    print(f"Deleted files: {file_v2_path}")
                else:
                    print(f"file does not exist: {file_v2_path}")

                if os.path.exists(file_v3_path):
                    os.remove(file_v3_path)
                    print(f"Deleted files: {file_v3_path}")
                else:
                    print(f"file does not exist: {file_v3_path}")

def compute_personalization_scores_from_two_json(local_json_path: str, aggregated_json_path: str):
    """
    Extract data item by item from two JSON files,
    calculate the personalized score (aggregated_perplexity / local_perplexity),
    and return:
    List[(Sample Index, Personalized Score)], sorted in descending order of the score.
    """
    with open(local_json_path, 'r', encoding='utf-8') as f:
        local_data = json.load(f)
    with open(aggregated_json_path, 'r', encoding='utf-8') as f:
        aggregated_data = json.load(f)

    scores = []
    for idx, (sample_local, sample_agg) in enumerate(zip(local_data, aggregated_data)):
        local = sample_local.get("local_perplexity")
        aggregated = sample_agg.get("aggregated_perplexity")
        if local is None or aggregated is None:
            continue
        score = aggregated / local if local > 0 else float('inf')
        scores.append((idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def map_scores_to_json_from_two_json(local_json_path: str, aggregated_json_path: str, output_json_path: str):
    """
    Sort the data based on the personalized scores calculated from the two JSON files.
    Merge the sample data from both files, sort them according to the scores,
    and save the sorted data as a new JSON file.

    Parameters:
    local_json_path: Path to the JSON file containing original data and local_perplexity
    aggregated_json_path: Path to the JSON file containing original data and aggregated_perplexity
    output_json_path: Path to save the sorting results
    """
    # Read data from two files
    with open(local_json_path, 'r', encoding='utf-8') as f:
        local_data = json.load(f)
    with open(aggregated_json_path, 'r', encoding='utf-8') as f:
        aggregated_data = json.load(f)

    # Merge sample data from both files, assuming the sample order in both files is consistent
    merged_data = []
    for sample_local, sample_agg in zip(local_data, aggregated_data):
        merged_sample = sample_local.copy()
        merged_sample.update(sample_agg)
        merged_data.append(merged_sample)

    # Calculate personalized scores and obtain sorting indices
    scores = compute_personalization_scores_from_two_json(local_json_path, aggregated_json_path)
    # Reconstruct the merged data list based on sorting indices
    sorted_data = [merged_data[idx] for idx, score in scores]

    # Create output directory if it doesn't exist
    dir_path = os.path.dirname(output_json_path)
    os.makedirs(dir_path, exist_ok=True)

    # Save the sorted data to a new JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_data, f, indent=4, ensure_ascii=False)

    print(f"Sorted data has been saved to {output_json_path}")



if __name__ == "__main__":
    fire.Fire(fl_finetune)
