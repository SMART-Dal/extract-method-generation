import json
import torch
import pandas as pd
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer, AutoTokenizer, AutoModelForCausalLM

def data_gen(path):
    with open(path, 'r') as f:
        for line in f:
            example = json.loads(line.strip())    
            yield example

def code_t5(file_path):
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    # model = T5ForConditionalGeneration.from_pretrained("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/refactoring-finetune/CodeT5/sh/saved_models/refactoring/codet5_small_all_lr5_bs32_src320_trg256_pat5_e100/checkpoint-best-bleu")
    model = T5ForConditionalGeneration.from_pretrained(file_path)

    model = model.to('cuda')

    ls_data = []
    for example in data_gen():
            tok_input = tokenizer(example["Smelly Sample"].replace("\t"," "),return_tensors="pt", padding="max_length", max_length=512, truncation=True).input_ids
            tok_input = tok_input.to('cuda')
            # print(model.generate(tok_input))
            # print(tok_input.size())
            with torch.no_grad():
                gen_tokens = model.generate(tok_input, max_length = 512)

            ft = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            ls_data.append([example["Smelly Sample"].replace("\t"," "),example["Method after Refactoring"].replace("\t"," ") + example["Extracted Method"].replace("\t"," "), ft])

    df = pd.DataFrame(ls_data,columns=['ss','gt','ft'])
    df.to_csv('output_rl.csv', index=False)


def code_llama(file_path, model_path):
    base_model = model_path
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    model.eval()
    for example in data_gen(file_path):

        full_prompt =f"""You are a powerful code refactoring model. Your job is to refactor the given method using extract method refactoring. 

                        ### To be-refacored code:
                        {example["Smelly Sample"]}

                        ### Refactored Code:
                        """        
        model_input = tokenizer(full_prompt, return_tensors="pt").to('cuda')
        # print(model_input.input_ids.size())
        with torch.no_grad():
            print(tokenizer.decode(model.generate(**model_input, max_new_tokens=3072)[0], skip_special_tokens=True).strip())
        break

    return 


if __name__=="__main__":
    code_llama("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/reinforcement-learning/test-data/test.jsonl","codellama/CodeLlama-7b-hf")