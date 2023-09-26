import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from pathlib import Path


model_path = Path('~/text-generation-webui/models/TheBloke_Upstage-Llama-2-70B-instruct-v2-GPTQ_gptq-4bit-32g-actorder_True')


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=False,
    revision="main"
)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

print('loaded model successfully!')

model_name = "TheBloke/Upstage-Llama-2-70B-instruct-v2-GPTQ"
# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=False,
    revision="gptq-4bit-32g-actorder_True"
)
