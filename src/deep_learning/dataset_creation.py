import os
import json
import random



def gen_data(folder_path):
    for file in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, file)):
            continue       
        with open(os.path.join(folder_path, file), 'r') as source_file:
            content = source_file.read()
            with open("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl/data.jsonl", "a") as destination_file:
                destination_file.write(content)

def create_train_test_split(jsonl_file_path, ouput_folder_path):
    with open(jsonl_file_path, 'r') as file:
        lines = file.readlines()

    random.shuffle(lines)

    train_idx = int(0.7 * len(lines))
    val_idx = train_idx + int(0.1 * len(lines))

    with open(os.path.join(ouput_folder_path,'train.jsonl'), 'w') as file:
        for line in lines[:train_idx]:
            file.write(line)

    with open(os.path.join(ouput_folder_path,'val.jsonl'), 'w') as file:
        for line in lines[train_idx:val_idx]:
            file.write(line)

    with open(os.path.join(ouput_folder_path,'test.jsonl'), 'w') as file:
        for line in lines[val_idx:]:
            file.write(line)

if __name__== "__main__":
    # gen_data("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/output/localscratch/ip1102.19990145.0/extract-method-generation/data/output/")
    # create_train_test_split("data/dl/data.jsonl", "data/dl/")
    print("Main")