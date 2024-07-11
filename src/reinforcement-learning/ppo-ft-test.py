from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, create_reference_model, PPOTrainer
from transformers import AutoTokenizer
import json
import torch

model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/reinforcement-learning/ppo-output")
tokenizer = AutoTokenizer.from_pretrained("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/reinforcement-learning/ppo-output")

with open("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/reinforcement-learning/test-data/val.jsonl",'r') as f:
    for line in f:
        ex = json.loads(line.strip())
        tok_input = tokenizer(ex["Smelly Sample"].replace("\t"," "),return_tensors="pt", padding="max_length", max_length=512, truncation=True).input_ids
        tok_input = tok_input.to('cuda')
        model = model.to('cuda')
        with torch.no_grad():
            gen_tokens = model.generate(tok_input, max_length = 512)
            # gen_tokens = model.generate(list(tok_input).unsqueeze(dim=0), max_length = 512)
        ft = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        print("Normal generation:")
        print(ft)
        break
