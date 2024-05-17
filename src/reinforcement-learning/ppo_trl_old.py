from dataclasses import dataclass, field
from typing import Optional

import torch
import json
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead, create_reference_model, set_seed
from trl.core import LengthSampler, respond_to_batch
from reward_test import get_reward

tqdm.pandas()

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="Salesforce/codet5-small", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="Salesforce/codet5-small", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default="./ppo-output",
        metadata={"help": "the path to save the model"},
    )

class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        with open(file_path, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                self.data.append((example["Smelly Sample"].replace("\t"," "), example["Method after Refactoring"].replace("\t"," ") + example["Extracted Method"].replace("\t"," ")))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, target = self.data[idx]
        inputs = self.tokenizer(features, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        labels = self.tokenizer(target, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        return {"input_ids": inputs["input_ids"].squeeze(0), "attention_mask": inputs["attention_mask"].squeeze(0), "labels": labels["input_ids"].squeeze(0)} 

def custom_collator(data):
    batch = {}
    for key in data[0].keys():
        # Collect elements as lists to preserve individual structure
        batch[key] = [sample[key] for sample in data]
    return batch


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
print(script_args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)

model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(script_args.model_name)

train_dataset = CustomDataset("./test-data/train.jsonl", tokenizer)
train_dataloader = DataLoader(train_dataset, 4, True)

val_dataset = CustomDataset("./test-data/val.jsonl", tokenizer)
val_dataloader = DataLoader(val_dataset, 4, True)

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=100,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    ratio_threshold=200
)

# # set seed before initializing value head for deterministic eval
set_seed(config.seed)

ref_model = create_reference_model(model)

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=train_dataset,
    data_collator=custom_collator,
    optimizer=optimizer
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 512
}
output_min_length = 20
output_max_length = 300
output_length_sampler = LengthSampler(output_min_length, output_max_length)

model_save_path = script_args.model_save_path

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]
    print("========================GEN===========================")
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response_ids"] = response_tensors
    rewards = []

    for query, response in zip(query_tensors, response_tensors):
        smelly_code_sample = tokenizer.decode(query.squeeze(), skip_special_tokens=True)
        refactored_code = tokenizer.decode(response.squeeze(), skip_special_tokens=True)
        reward = get_reward(smelly_code_sample, refactored_code)
        rewards.append(torch.tensor(reward))

    assert len(query_tensors) == len(response_tensors) == len(rewards)

    # # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["input_ids","response_ids"])

    # Save model every 100 epochs
    if epoch % 10 == 0:
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.save_pretrained(model_save_path)

ppo_trainer.save_pretrained(model_save_path)
print("End of program")