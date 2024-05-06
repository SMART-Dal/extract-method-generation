import json
import torch
import pandas as pd
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
model = T5ForConditionalGeneration.from_pretrained("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/refactoring-finetune/CodeT5/sh/saved_models/refactoring/codet5_small_all_lr5_bs32_src320_trg256_pat5_e100/checkpoint-best-bleu")

model = model.to('cuda')

ls_data = []
with open("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/reinforcement-learning/test-data/test.jsonl", 'r') as f:
    for line in f:
        example = json.loads(line.strip())
        tok_input = tokenizer(example["Smelly Sample"].replace("\t"," "),return_tensors="pt", padding="max_length", max_length=512, truncation=True).input_ids
        tok_input = tok_input.to('cuda')
        # print(model.generate(tok_input))
        # print(tok_input.size())
        with torch.no_grad():
            gen_tokens = model.generate(tok_input, max_length = 512)

        ft = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        ls_data.append([example["Smelly Sample"].replace("\t"," "),example["Method after Refactoring"].replace("\t"," ") + example["Extracted Method"].replace("\t"," "), ft])

df = pd.DataFrame(ls_data,columns=['ss','gt','ft'])
df.to_csv('output.csv', index=False)