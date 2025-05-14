from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
import torch
from datasets import Dataset
import pandas as pd
import numpy as np


df = pd.read_csv('accepted_rejected.csv')

prompts = []
selected = []
rejected = []


for index, row in df.iterrows():
    prompt = row['prompt']
    accepted = row['accepted']
    rejected = row['rejected']

    prompts.append(prompt)
    selected.append(accepted)
    rejected.append(rejected)

dpo_dataset_dict = {
    "prompt": prompts,
    "chosen": selected,
    "rejected": rejected,
}
del df
dataset = Dataset.from_dict(dpo_dataset_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")           
model = AutoModelForCausalLM.from_pretrained(               
    "gpt2",
    state_dict=torch.load("sft_gpt.bin"),
).to(device)

training_args = TrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    gradient_checkpointing =True,
    max_grad_norm= 0.3,
    num_train_epochs=3,
    save_steps= 100,
    learning_rate=2e-4,
    # bf16=False,
    save_total_limit=3,
    logging_steps=10,
    output_dir="DPOP",
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    remove_unused_columns=False
)


dpo_trainer = DPOTrainer(
    model,
    # ref_model,
    args=training_args,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_prompt_length=128,
    max_length=256,
)

dpo_trainer.train()

dpo_trainer.save_model("new_gpt")
