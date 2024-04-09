import torch
from tqdm import tqdm
import pandas as pd
import json

tqdm.pandas()

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

# from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
# from trl.core import LengthSampler

# config = PPOConfig(
#     model_name="lvwerra/gpt2-imdb",
#     learning_rate=1.41e-5,
#     log_with="wandb",
# )

# sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}


def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)
    print(ds.features)
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    print(ds[0:5])
    return ds


# dataset = build_dataset(config)

# model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
# ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
# tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# tokenizer.pad_token = tokenizer.eos_token

# def collator(data):
#     return dict((key, [d[key] for d in data]) for key in data[0])

# ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# device = ppo_trainer.accelerator.device
# if ppo_trainer.accelerator.num_processes == 1:
#     device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
# sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)


# text = "this movie was really bad!!"
# sentiment_pipe(text, **sent_kwargs)

# output_min_length = 4
# output_max_length = 16
# output_length_sampler = LengthSampler(output_min_length, output_max_length)


# generation_kwargs = {
#     "min_length": -1,
#     "top_k": 0.0,
#     "top_p": 1.0,
#     "do_sample": True,
#     "pad_token_id": tokenizer.eos_token_id,
# }


# for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
#     query_tensors = batch["input_ids"]

#     #### Get response from gpt2
#     response_tensors = []
#     for query in query_tensors:
#         gen_len = output_length_sampler()
#         generation_kwargs["max_new_tokens"] = gen_len
#         response = ppo_trainer.generate(query, **generation_kwargs)
#         response_tensors.append(response.squeeze()[-gen_len:])
#     batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

#     #### Compute sentiment score
#     texts = [q + r for q, r in zip(batch["query"], batch["response"])]
#     pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
#     rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

#     #### Run PPO step
#     stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
#     ppo_trainer.log_stats(stats, batch, rewards)

def test():
    model = AutoModelForSeq2SeqLM.from_pretrained("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/CodeT5/CodeT5/sh/saved_models/refactoring/codet5_small_all_lr5_bs32_src320_trg256_pat5_e100/checkpoint-best-bleu")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

    with open("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context/val.jsonl") as f:
        lines = f.readlines()
        for line in lines:
            d = json.loads(line)
            break
        print(d["Smelly Sample"])

test()