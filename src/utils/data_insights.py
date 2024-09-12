import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
    )

# with open("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-large/preprocessed/train.jsonl", "+r") as f:
#     lines = f.read()
#     for line in lines:
#         example = json.loads(line)
#         print(example["Input"])

def model_details(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    print(model.config)

def dec_only_model_details(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    print(model.config)

model_details("Salesforce/codet5-base")
# model_details("uclanlp/plbart-refine-java-small")
# dec_only_model_details("microsoft/CodeGPT-small-java")
