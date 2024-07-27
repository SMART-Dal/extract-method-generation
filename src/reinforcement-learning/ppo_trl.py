from dataclasses import dataclass, field
from typing import Optional

import torch
import json
import datetime
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
    DataCollatorForSeq2Seq,
    DefaultDataCollator
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead, create_reference_model, set_seed
from trl.core import LengthSampler, respond_to_batch
from reward_test import Reward

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
    train_data_file_path: Optional[str] = field(default=None,metadata={"help":"the path to the train data file"})
    eval_data_file_path: Optional[str] = field(default=None,metadata={"help":"the path to the eval data file"})
    test_data_file_path: Optional[str] = field(default=None,metadata={"help":"the path to the eval data file"})    

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

train_dataset = load_dataset("json",data_files=script_args.train_data_file_path, split="train")
eval_dataset = load_dataset("json",data_files=script_args.eval_data_file_path, split="train")

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(script_args.model_name)

print(next(model.parameters()).is_cuda)

def preprocess_function(examples):
    inputs = examples["Input"]
    targets = examples["Output"]

    padding = "max_length"
    # inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=512, padding=padding, truncation=True, return_tensors="pt")
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=512, padding=padding, truncation=True, return_tensors="pt")

    # if padding == "max_length" and data_args.ignore_pad_token_for_loss:
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    model_inputs['input'] = inputs
    model_inputs['output'] = targets
    return model_inputs

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    desc="Running tokenizer on train dataset",
)

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    desc="Running tokenizer on eval dataset"
)

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=100,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    # ratio_threshold=200,
    early_stopping=True
)
ref_model = create_reference_model(model)
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    # pad_to_multiple_of=8 if training_args.fp16 else None,
)

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    optimizer=optimizer
)

generation_kwargs = {
    "min_length": -1,
    # "top_k": 0.0,
    # "top_p": 1.0,
    # "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 512
}
output_min_length = 512
output_max_length = 512
output_length_sampler = LengthSampler(output_min_length, output_max_length)

model_save_path = script_args.model_save_path

train_dataloader = DataLoader(train_dataset, script_args.batch_size, True, collate_fn=DefaultDataCollator())

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels
# for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
for epoch, batch in tqdm(enumerate(train_dataloader)):
    # print(len(batch["input_ids"]))
    query_tensors = batch["input_ids"]
    query_tensors = query_tensors.to('cuda')
    # print("========================GEN===========================")
    response_tensors = []
    for query in query_tensors:
        # gen_len = output_length_sampler()
        gen_len = 512
        # generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        # response = ppo_trainer.generate(query, max_length = 512)
        response_tensors.append(response.squeeze()[-gen_len:])
       
    batch["response_ids"] = response_tensors
    
    # print("========================GEN PPO===========================")
    with open(f"./trl_intermediate_logs/logs_{str(datetime.datetime.now().date())}_{str(datetime.datetime.now().time())}.txt", "a+") as f:
        rewards = []
        for query, response in zip(query_tensors, response_tensors):
            smelly_code_sample = tokenizer.decode(query.squeeze(), skip_special_tokens=True)
            refactored_code = tokenizer.decode(response.squeeze(), skip_special_tokens=True)
            reward = Reward().get_reward(smelly_code_sample, refactored_code)
            rewards.append(torch.tensor(reward))
        
            f.write("\nSmelly Code:"+"\n"+smelly_code_sample)
            f.write("\nRefactored Code:"+"\n"+refactored_code)
            f.write("\nReward:"+"\n"+str(reward))
            
        f.write("\n============End of batch================\n")

    assert len(query_tensors) == len(response_tensors) == len(rewards)

    # print("Type of query tensors:",type(query_tensors))
    # print("Shape of query tensors:", query_tensors.shape)

    query_tensors = list(torch.unbind(query_tensors, dim=0)) # ppo_trainer.step only takes list of tensors

    # print("Test:")
    # print(tokenizer.decode(query_tensors[0].squeeze(), skip_special_tokens=True))

    # # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["input_ids","response_ids"])

    # Save model every 100 epochs
    if epoch % 10 == 0:
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.save_pretrained(model_save_path)

ppo_trainer.save_pretrained(model_save_path)
print("End of program")

"""
python ppo_trl.py 
--model_name /home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/refactoring-finetune/ft-scripts/output/code-t5-fine-tuned 
--tokenizer_name /home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/refactoring-finetune/ft-scripts/output/code-t5-fine-tuned 
--log_with wandb 
--train_data_file_path /home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context-len/train.jsonl 
--eval_data_file_path /home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context-len/val.jsonl
"""

"""
python ppo_trl.py 
--model_name /home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/refactoring-finetune/ft-scripts/output/code-t5-fine-tuned 
--tokenizer_name /home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/refactoring-finetune/ft-scripts/output/code-t5-fine-tuned 
--log_with wandb 
--train_data_file_path /home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-large/preprocessed/train.jsonl 
--eval_data_file_path /home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-large/preprocessed/val.jsonl 
"""