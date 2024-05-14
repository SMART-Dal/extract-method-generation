import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_trg_col_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    avg_trg_collated_len_tokenize = []
    for ex in tqdm(examples):
        if is_tokenize:
            avg_src_len.append(len(str(ex["Smelly Sample"]).split()))
            avg_trg_len.append(len(str(ex["Method after Refactoring"]).split()))
            avg_trg_col_len.append(len(str(ex["Method after Refactoring"]).split())+len(str(ex['Extracted Method']).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex["Smelly Sample"])))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex["Method after Refactoring"]))))
            avg_trg_collated_len_tokenize.append(len(tokenizer.tokenize(str(ex["Method after Refactoring"]+str(ex['Extracted Method'])))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        print("Read %d examples, avg src len: %d, avg trg len: %d, avg trg col len: %d, max src len: %d, max trg len: %d, max trg len col: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), np.mean(avg_trg_col_len), max(avg_src_len), max(avg_trg_len), max(avg_trg_col_len))
        print("[TOKENIZE] avg src len: %d, avg trg len: %d, avg trg len col: %d, max src len: %d, max trg len: %d, max trg len col: %d", 
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), np.mean(avg_trg_collated_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize), max(avg_trg_collated_len_tokenize))
    else:
        print("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        

if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    # model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
    calc_stats(load_dataset("json",
                            data_files="/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context/train.jsonl",
                            split='train'),
                tokenizer,
                True
               )