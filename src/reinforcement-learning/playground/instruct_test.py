import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
model = model.half().cuda()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
ls_data = []
with open("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/reinforcement-learning/test-data/test.jsonl", 'r') as f:
    for line in f:
        example = json.loads(line.strip())
        # tokenizer.pad_token = tokenizer.eos_token
        # tok_input = tokenizer("Perform extract method refactoring on the following code: \n\n"+example["Smelly Sample"].replace("\t"," "),
        #                       return_tensors="pt", 
        #                       max_length=1024,
        #                       truncation=True)
        prompt = "Perform extract method refactoring on the following code: \n\n"+example["Smelly Sample"].replace("\t"," ")
        tok_input = tokenizer(prompt, return_tensors="pt", max_length=2048).to('cuda')
        # tok_input = tok_input
        # print(tok_input.input_ids.tolist()[0])
        # print(tokenizer.decode(tok_input.input_ids.tolist()[0],skip_special_tokens=True))

        gen_tokens = model.generate(tok_input.input_ids, max_length=2048)

        ft = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        ls_data.append([example["Smelly Sample"].replace("\t"," "),example["Method after Refactoring"].replace("\t"," ") + example["Extracted Method"].replace("\t"," "), ft])

df = pd.DataFrame(ls_data,columns=['ss','gt','ft'])
df.to_csv('output_instruct.csv', index=False)

# model = AutoModelForCausalLM.from_pretrained("gpt2")
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = model.half().cuda()

# prompt = "GPT2 is a model developed by OpenAI."

# input_ids = tokenizer(prompt, return_tensors="pt").to('cuda')

# gen_tokens = model.generate(
#     input_ids.input_ids,
#     do_sample=True,
#     temperature=0.9,
#     max_length=100,
# )
# gen_text = tokenizer.batch_decode(gen_tokens)[0]
