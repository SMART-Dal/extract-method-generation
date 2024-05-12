import numpy as np
import os
import json
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def generate_modified_data(dataset, tokenizer, file_name):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(project_root, "data", "dl-no-context-len")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{file_name}.jsonl'), 'w') as f:
        for data in tqdm(dataset):
            if len(tokenizer.tokenize(data["Smelly Sample"])) > 512:
                continue
            if len(tokenizer.tokenize(str(data["Method after Refactoring"])+str(data['Extracted Method']))) > 512:
                continue
            sample = {"Input": str(data["Smelly Sample"]), "Output": str(data["Method after Refactoring"])+"\n"+str(data['Extracted Method'])}
            f.write(json.dumps(sample) + '\n')

def calc_stats(examples, tokenizer=None, collation=False):
    avg_src_len = []
    avg_trg_len = []
    avg_trg_col_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    avg_trg_collated_len_tokenize = []
    src_count, trg_count, trg_col_count = 0,0,0
    if collation:
        keys = ("Smelly Sample", "Method after Refactoring", 'Extracted Method')
    else:
        keys = ("Input", "Output")   
    for ex in tqdm(examples):

        avg_src_len.append(len(str(ex[keys[0]]).split()))
        avg_trg_len.append(len(str(ex[keys[1]]).split()))
        if collation:
            avg_trg_col_len.append(len(str(ex[keys[1]]).split())+len(str(ex[keys[2]]).split()))
        
        tmp = len(tokenizer.tokenize(ex[keys[0]]))
        if tmp>512:
            src_count+=1
        avg_src_len_tokenize.append(tmp)
        
        tmp = len(tokenizer.tokenize(str(ex[keys[1]])))
        if tmp>512:
            trg_count+=1
        avg_trg_len_tokenize.append(tmp)
        
        if collation:
            tmp = len(tokenizer.tokenize(str(ex[keys[1]]+str(ex[keys[2]]))))
            if tmp>512:
                trg_col_count+=1
            avg_trg_collated_len_tokenize.append(tmp)

        # avg_src_len.append(len(ex.source.split()))
        # avg_trg_len.append(len(str(ex.target).split()))


    print(f"Read {len(examples)} examples \navg src len: {np.mean(avg_src_len)}\n avg trg len: {np.mean(avg_trg_len)}\n avg trg col len: {np.mean(avg_trg_col_len)} \
            \n max src len: {max(avg_src_len)}\n max trg len: {max(avg_trg_len)}\n max trg len col: {max(avg_trg_col_len) if avg_trg_col_len else None }")
    print(f"[TOKENIZE] avg src len: {np.mean(avg_src_len_tokenize)}\n avg trg len: {np.mean(avg_trg_len_tokenize)}\n avg trg len col: {np.mean(avg_trg_collated_len_tokenize) if avg_trg_collated_len_tokenize else None}\n max src len: {max(avg_src_len_tokenize)}\n max trg len: {max(avg_trg_len_tokenize)}\n max trg len col: {max(avg_trg_collated_len_tokenize) if avg_trg_collated_len_tokenize else None}")
    print(f"[TOKENIZE] src count above model max: {src_count} \n target count above model max: {trg_count}\n target col above model max: {trg_col_count}")
        # print("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
        #             len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))

def calc_stats_mod_method(examples, tokenizer=None):
    avg_src_len = []
    avg_trg_len = []
    avg_trg_col_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    avg_trg_collated_len_tokenize = []
    src_count, trg_count, trg_col_count = 0,0,0
    for ex in tqdm(examples):

        avg_src_len.append(len(str(ex["Input"]).split()))
        avg_trg_len.append(len(str(ex["Output"]).split()))
        # avg_trg_col_len.append(len(str(ex["Output"]).split())+len(str(ex['Extracted Method']).split()))
        
        tmp = len(tokenizer.tokenize(ex["Input"]))
        if tmp>512:
            src_count+=1
        avg_src_len_tokenize.append(tmp)
        
        tmp = len(tokenizer.tokenize(str(ex["Output"])))
        if tmp>512:
            trg_count+=1
        avg_trg_len_tokenize.append(tmp)
        
        # tmp = len(tokenizer.tokenize(str(ex["Output"]+str(ex['Extracted Method']))))
        # if tmp>512:
        #     trg_col_count+=1
        #     avg_trg_collated_len_tokenize.append(tmp)

        # avg_src_len.append(len(ex.source.split()))
        # avg_trg_len.append(len(str(ex.target).split()))


    print(f"Read {len(examples)} examples, avg src len: {np.mean(avg_src_len)}, avg trg len: {np.mean(avg_trg_len)}, max src len: {max(avg_src_len)}, max trg len: {max(avg_trg_len)}")
    print(f"[TOKENIZE] avg src len: {np.mean(avg_src_len_tokenize)}, avg trg len: {np.mean(avg_trg_len_tokenize)}, max src len: {max(avg_src_len_tokenize)}, max trg len: {max(avg_trg_len_tokenize)}")
    print(f"[TOKENIZE] src count above model max: {src_count}, target count above model max: {trg_count}")
    print(f"Read {len(examples)} examples, avg src len: {np.mean(avg_src_len)}, avg trg len: {np.mean(avg_trg_len)}, max src len: {max(avg_src_len)}, max trg len: {max(avg_trg_len)}")


if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    # # model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
    type = sys.argv[1]
    if type == "calc":
        data_files = sys.argv[2]
        calc_stats(load_dataset("json",
                                data_files=data_files,
                                split='train'),
                    tokenizer
                )
        # calc_stats_mod_method(load_dataset(
        #     "json",
        #     data_files="/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context-len/val.jsonl",
        #     split="train",
        # ),
        # tokenizer)        
    elif type == "generate":
        generate_modified_data(load_dataset("json",
                                data_files="/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context/val.jsonl",
                                split='train'),
                                tokenizer,
                                "val"
                                )
    else:
        print("Wrong Choice")        


    
    