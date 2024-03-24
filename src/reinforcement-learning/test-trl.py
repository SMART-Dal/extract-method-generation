# # imports
# import torch
# from transformers import AutoTokenizer
# from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
# from trl.core import respond_to_batch

# # get models
# model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
# model_ref = create_reference_model(model)
# model = model.half().cuda()
# model_ref = model_ref.half().cuda()

# tokenizer = AutoTokenizer.from_pretrained('gpt2')
# tokenizer.pad_token = tokenizer.eos_token

# # initialize trainer
# ppo_config = PPOConfig(batch_size=1, mini_batch_size=1)

# print(ppo_config)

# # encode a query
# query_txt = "This morning I went to the "
# query_tensor = tokenizer.encode(query_txt, return_tensors="pt")
# query_tensor = query_tensor.to("cuda")

# # get model response
# response_tensor  = respond_to_batch(model, query_tensor)

# print(tokenizer.decode(response_tensor[0]))

# # create a ppo trainer
# ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

# # define a reward for response
# # (this could be any reward such as human feedback or output from another model)
# reward = [torch.tensor(1.0)]

# # train model for one step with ppo
# train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
# # ppo_trainer.log_stats(train_stats, )


# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    AutoModelForSeq2SeqLM
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead, create_reference_model, set_seed
from trl.core import LengthSampler, respond_to_batch


tqdm.pandas()

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPTJ model to generate less toxic contents
# by using allenai/real-toxicity-prompts dataset. We use PPO
#  (proximal policy optimization) to optimize the model.
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `project_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="ybelkada/gpt-j-6b-sharded-bf16", metadata={"help": "the model name"})
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
        # return {"input_ids": inputs.input_ids.squeeze(0), "attention_mask": inputs.attention_mask, "labels": labels.input_ids}
        return {"input_ids": inputs["input_ids"].squeeze(0), "attention_mask": inputs["attention_mask"].squeeze(0), "labels": labels["input_ids"].squeeze(0)}
        # return inputs, labels    

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
print(script_args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_name)

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
)


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
# def build_dataset(
#     config, dataset_name="allenai/real-toxicity-prompts", input_min_text_length=5, input_max_text_length=10
# ):
#     """
#     Build dataset for training. This builds the dataset from `load_dataset`, one should
#     customize this function to train the model on its own dataset.

#     Args:
#         dataset_name (`str`):
#             The name of the dataset to be loaded.

#     Returns:
#         dataloader (`torch.utils.data.DataLoader`):
#             The dataloader for the dataset.
#     """
#     tokenizer = AutoTokenizer.from_pretrained(config.model_name)
#     tokenizer.pad_token = tokenizer.eos_token

#     ds = load_dataset(dataset_name, split="train")

#     def filter_fn(sample):
#         toxicity = sample["prompt"]["toxicity"]
#         return toxicity is not None and toxicity > 0.3

#     ds = ds.filter(filter_fn, batched=False)

#     input_size = LengthSampler(input_min_text_length, input_max_text_length)

#     def tokenize(sample):
#         prompt = sample["prompt"]["text"]
#         continuation = sample["continuation"]["text"]

#         sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size()]
#         sample["query"] = tokenizer.decode(sample["input_ids"])
#         return sample

#     ds = ds.map(tokenize, batched=False)
#     ds.set_format(type="torch")

#     ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]

#     return ds


# # We retrieve the dataloader by calling the `build_dataset` function.
# min_input_length = 30
# max_input_length = 40
# dataset = build_dataset(config, input_min_text_length=min_input_length, input_max_text_length=max_input_length)

# def collator(data):
#     return {key: [d[key] for d in data] for key in data[0]}


# # set seed before initializing value head for deterministic eval
set_seed(config.seed)

# # Now let's build the model, the reference model, and the tokenizer. We first load the model
# # in bfloat16 to save memory using `transformers`.
# model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
# # And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`.
# model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

# # We create a reference model by sharing 20 layers
ref_model = create_reference_model(model)

# # We make sure to use `Adam` optimizer on the model parameters that require gradients.
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

# # GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# # only for this model.
# tokenizer = AutoTokenizer.from_pretrained(config.model_name)
# tokenizer.pad_token = tokenizer.eos_token

# # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=train_dataset,
    # data_collator=collator,
    optimizer=optimizer,
)

# # We then build the reward pipeline, we will use the toxicity model to compute the reward.
# # We first load the toxicity model and tokenizer.
# toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
# toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
# # We load the toxicity model in fp16 to save memory.
# toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16).to(
#     ppo_trainer.accelerator.device
# )


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
# output_min_length = 20
# output_max_length = 30
# output_length_sampler = LengthSampler(output_min_length, output_max_length)

# model_save_path = script_args.model_save_path

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]
    # print(query_tensors)
    # reponse=respond_to_batch(model, query_tensors)
    # print(reponse)
    for query in query_tensors:
        reponse = ppo_trainer.generate(query, **generation_kwargs)
        print(tokenizer.decode(reponse.squeeze(), skip_special_tokens=True))
    break
    # Get response from the policy model
    # response_tensors = []
    # for query in query_tensors:
        # gen_len = output_length_sampler()
        # generation_kwargs["max_new_tokens"] = gen_len
        # response = ppo_trainer.generate(query, **generation_kwargs)
        # response_tensors.append(response.squeeze()[-gen_len:])
    # batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute sentiment score
    # texts = batch["response"]
    # toxicity_inputs = toxicity_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
    #     ppo_trainer.accelerator.device
    # )
    # logits = toxicity_model(**toxicity_inputs).logits.float()
    # toxicity_labels = (logits[:, 0]).tolist()

    # rewards = [torch.tensor(output) for output in toxicity_labels]

    # # Run PPO step
    # stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    # ppo_trainer.log_stats(stats, batch, rewards)

    # # Save model every 100 epochs
    # if epoch % 100 == 0:
    #     if ppo_trainer.accelerator.is_main_process:
    #         ppo_trainer.save_pretrained(model_save_path)