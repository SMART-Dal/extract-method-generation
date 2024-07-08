import json
import torch
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, create_reference_model, PPOTrainer
from transformers import AutoTokenizer
from torch.optim import Adam


model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/refactoring-finetune/ft-scripts/output")
tokenizer = AutoTokenizer.from_pretrained("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/refactoring-finetune/ft-scripts/output")


config = PPOConfig(
    model_name="/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/refactoring-finetune/ft-scripts/output",
    learning_rate=0.001,
    ppo_epochs=100,
    mini_batch_size=4,
    batch_size=8,
    gradient_accumulation_steps=1,
    ratio_threshold=200,
    early_stopping=True
)
ref_model = create_reference_model(model)
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    optimizer=Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 512
}

with open("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/reinforcement-learning/test-data/val.jsonl",'r') as f:
    for line in f:
        ex = json.loads(line.strip())
        tok_input = tokenizer(ex["Smelly Sample"].replace("\t"," "),return_tensors="pt", padding="max_length", max_length=512, truncation=True).input_ids
        tok_input = tok_input.to('cuda')
        model = model.to('cuda')
        with torch.no_grad():
            # gen_tokens = model.generate(tok_input, max_length = 512)
            gen_tokens = model.generate(list(tok_input).unsqueeze(dim=0), max_length = 512)
        ft = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        print("Normal generation:")
        print(ft)
        print("===============================================")
        print("PPO generation")
        gen_len = 512
        generation_kwargs["max_new_tokens"] = gen_len
        # Converting to list is a generic practice: 
        response = ppo_trainer.generate(list(tok_input), **generation_kwargs)
        ft_ppo = tokenizer.decode(response[0], skip_special_tokens=True)
        print(ft_ppo)
        break
