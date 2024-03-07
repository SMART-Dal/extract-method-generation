import os
import random
import sys
from tqdm import tqdm

def gen_data(folder_path, data_file):
    for file in tqdm(os.listdir(folder_path)):
        if os.path.isdir(os.path.join(folder_path, file)):
            continue       
        with open(os.path.join(folder_path, file), 'r') as source_file:
            content = source_file.read()
            with open(data_file, "a") as destination_file:
                destination_file.write(content)

def create_train_test_split(jsonl_file_path, ouput_folder_path):
    with open(jsonl_file_path, 'r') as file:
        lines = file.readlines()

    print("Total refactoring instances: ", len(lines))
    random.shuffle(lines)

    train_idx = int(0.7 * len(lines))
    val_idx = train_idx + int(0.1 * len(lines))

    print("Creating train jsonl...")
    with open(os.path.join(ouput_folder_path,'train.jsonl'), 'w') as file:
        for line in tqdm(lines[:train_idx]):
            file.write(line)
    
    print("Creating val jsonl...")
    with open(os.path.join(ouput_folder_path,'val.jsonl'), 'w') as file:
        for line in tqdm(lines[train_idx:val_idx]):
            file.write(line)

    print("Creating test jsonl...")
    with open(os.path.join(ouput_folder_path,'test.jsonl'), 'w') as file:
        for line in tqdm(lines[val_idx:]):
            file.write(line)

if __name__== "__main__":
    mode = sys.argv[1]

    if mode == "generate":
        input_folder = sys.argv[2]
        output_file_path = sys.argv[3]
        gen_data(input_folder, output_file_path)
        # gen_data("data/output/localscratch/ip1102.19990145.0/extract-method-generation/data/output/", "data/dl/data.jsonl")
    elif mode == "split":
        input_file_path = sys.argv[2]
        output_folder_path = sys.argv[3]
        create_train_test_split(input_file_path, output_folder_path)
        # create_train_test_split("data/dl/data.jsonl", "data/dl/")
    else:
        print("Mode can be either generate or split")