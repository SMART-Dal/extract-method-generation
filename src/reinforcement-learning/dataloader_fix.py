from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, DefaultDataCollator
from torch.utils.data import Dataset, DataLoader
import numpy as np

train_dataset = load_dataset("json",data_files="/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context-len/train.jsonl", split="train")
tokenizer = AutoTokenizer.from_pretrained("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/refactoring-finetune/ft-scripts/output")
model = AutoModelForSeq2SeqLM.from_pretrained("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/src/refactoring-finetune/ft-scripts/output")

def preprocess_function(examples):
    inputs = examples["Input"]
    targets = examples["Output"]    

    padding = "max_length"
    # inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=512, padding=padding, truncation=True, return_tensors="pt")
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=512, padding=padding, truncation=True, return_tensors="pt")

    # if padding == "max_length" and data_args.ignore_pad_token_for_loss:
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    
    # with open('ii.txt', 'a') as f:
    #     f.write("Sample One:\n")
    #     # f.write(str(model_inputs['input_ids'].size()))
    #     # f.write("\n")
    #     # f.write(model_inputs['input_ids'])
    #     f.write(np.array2string(model_inputs['input_ids'].numpy()[0]))
    #     f.write(np.array2string(tokenizer(inputs[0], padding="max_length", truncation=True, return_tensors="pt")['input_ids'].numpy()))

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    desc="Running tokenizer on train dataset",
)


label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    # pad_to_multiple_of=8 if training_args.fp16 else None,
)

train_dataloader = DataLoader(train_dataset, 4, False, collate_fn=DefaultDataCollator())
# train_dataloader = DataLoader(train_dataset, 4, False, collate_fn=data_collator)

for epoch, sample in enumerate(train_dataloader):
    print(sample.keys())
    print(sample['input_ids'].size())
    print(sample['input'])
    print(sample['output'])
    # print(sample['Input'][0])
    # print(sample['Output'][0])
    # print(sample['input_ids'][0])
    break

