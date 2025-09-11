import os
import gc
import fire
import gradio as gr
import torch
import transformers
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from utils.prompter import Prompter

# Determine the device to use
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


class EvalDataset(Dataset):
    def __init__(self, file, prompter, tokenizer, max_len=512):
        self.prompter = prompter
        self.tokenizer = tokenizer
        with open(file, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line.strip()) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ques = self.data[idx]
        sample = ques['instruction']
        prompt = self.prompter.generate_prompt(sample, None)
        return prompt, sample


def writeFile(s, path):
    dir_path = os.path.dirname(path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    with open(path, 'a+', encoding='utf-8') as f1:
        f1.write(s + '\n')


def main(
        # Model/data parameters
        load_8bit: bool = False,
        base_model: str = "./alpaca-7b-native",
        lora_weights_path: str = "./ablation/baseline/sample_4/40/19/local_output_32",
        lora_config_path: str = "./ablation/baseline/sample_4/40/19/local_output_32",
        prompt_template: str = "alpaca",

        # File parameters
        output_file: str = "/root/autodl-tmp/test1/ablation-prediction/FedIT-FT/sample_4/client_32/prediction_6.jsonl",  # 修改为 .jsonl
        test_file: str = "./data_train/initialized_data/8/local_test_6.jsonl",
):

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    # Step 1: Load base model
    if not lora_weights_path.endswith(".bin"):
        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
                vocab_size=32001
            )
        elif device == "mps":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
                vocab_size=32001
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True, vocab_size=32001
            )

        model.resize_token_embeddings(32001)

        model = PeftModel.from_pretrained(
            model,
            lora_weights_path,
            torch_dtype=torch.float16,
        )

    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            vocab_size=32001
        )

        model.resize_token_embeddings(32001)

        model = prepare_model_for_int8_training(model)
        config = LoraConfig.from_pretrained(lora_config_path)
        lora_weights = torch.load(lora_weights_path, map_location=lambda storage, loc: storage.cuda(0))
        model = PeftModel(model, config)
        set_peft_model_state_dict(model, lora_weights, "default")
        model.set_adapter('default')
        del lora_weights

    # Set tokenizer and model config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    tokenizer.padding_side = "left"
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # Seems to fix bugs for some users.

    model.eval()

    def evaluate(instruction, input=None, temperature=0.95, top_p=0.75, top_k=80, num_beams=4, max_new_tokens=80,
                 input_ids=None, **kwargs):
        if input_ids is not None:
            input_ids = input_ids.to(device)
        else:
            prompt = prompter.generate_prompt(instruction, input)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
                pad_to_max_length=True
            )

            input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            do_sample=True
        )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        ans = prompter.get_response(output)
        return ans

    save = output_file

    lines = open(test_file, 'r', encoding='utf-8').readlines()
    count = 0
    for i, line in enumerate(lines):
        line = line.strip()
        ques = json.loads(line)

        instruction = ques['instruction']
        context = ques.get('context', None)

        res = evaluate(instruction, input=context)

        tmp = {}
        tmp['instruction'] = instruction
        tmp['context'] = context
        tmp['answer'] = res
        tmp['category'] = ques.get('category', '')

        writeFile(json.dumps(tmp, ensure_ascii=False), save)

        count += 1
        print('num:', count)
        print("Instruction:", instruction)
        print("Context:", context)
        print("Response:", res)
        print("*****************************************************")

if __name__ == "__main__":

    epoch_set = []
    for client_id in range(0, 8):
        print(f"\n========= Evaluating Client {client_id} =========")
        fire.Fire(lambda: main(
            base_model='/root/autodl-tmp/test1/alpaca-7b-native',   # base_model
            lora_weights_path=f"/root/autodl-tmp/test1/lora-personalization/8/client_{client_id}/epoch_{epoch_set[client_id]}",
            lora_config_path=f"/root/autodl-tmp/test1/lora-personalization/8/client_{client_id}/epoch_{epoch_set[client_id]}",
            prompt_template="alpaca",   # Template
            output_file=f"/root/autodl-tmp/test1/method/prediction/our/prediction_{client_id}_0.jsonl",
            test_file=f"/root/autodl-tmp/test1/data/8/local_test_{client_id}.jsonl",
        ))
        gc.collect()
        torch.cuda.empty_cache()


