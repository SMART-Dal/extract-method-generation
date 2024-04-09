# import os
# import random
# from functools import lru_cache
import torch
import json
import nltk
from torch.utils.data import Dataset, DataLoader
from datasets import load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
# from datasets import load_dataset, DatasetDict, concatenate_datasets, Value, Sequence, ClassLabel
import numpy as np
from evaluate import load
nltk.download('punkt')
# import evaluate

# # Set seeds for reproducibility
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device being used: {device}")

# # Load data from JSONL files
# def read_jsonl(filepath):
#     dataset = []
#     with open(filepath, 'r') as f:
#         for line in f:
#             sample = eval(line)
#             dataset.append(sample)
#     return dataset

# train_data = read_jsonl("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context/train.jsonl")
# test_data = read_jsonl("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context/test.jsonl")
# val_data = read_jsonl("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context/val.jsonl")

# # Prepare datasets
# class CustomDataset(Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings

#     def __getitem__(self, idx):
#         item = {}
#         for key, tensor in self.encodings.items():
#             item[key] = tensor[idx].to(device)
#         return item

# tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

# train_encodings = tokenizer(list([s["Smelly Sample"] for s in train_data]), truncation=True, padding="longest", max_length=512)
# test_encodings = tokenizer(list([s["Smelly Sample"] for s in test_data]), truncation=True, padding="longest", max_length=512)
# val_encodings = tokenizer(list([s["Smelly Sample"] for s in val_data]), truncation=True, padding="longest", max_length=512)

# train_dataset = CustomDataset(train_encodings)
# eval_dataset = DatasetDict({'validation': CustomDataset(val_encodings), 'test': CustomDataset(test_encodings)})

# # Metrics
# metrics = evaluate.load("bleu", module_type="metric")
# metrics = evaluate.load("rouge", module_type="metric")
# metrics = evaluate.load("meteor", module_type="metric")

# # Model configuration and initialization
# model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small").to(device)

# # Training arguments
# training_args = Seq2SeqTrainingArguments(
#     output_dir='./results',          # output directory
#     num_train_epochs=3,              # total number of training epochs
#     per_device_train_batch_size=8,   # batch size per device during training
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
#     logging_steps=10,
#     evaluation_strategy='steps',     # evaluation strategy to use ("no", "steps", "epoch")
#     eval_steps=500,                   # Number of update steps between evaluations.
#     load_best_model_at_end=True,     # Whether or not to load the best model at the end of training
#     metric_for_best_model='bleu',      # metric to use to compare models
#     greater_is_better=True,         # whether higher is better for the selected metric
#     dataloader_num_workers=2,       # number of workers for loading data
#     fp16=False,                      # enable mixed precision training
#     run_name='CodeT5-smellfix',      # name of this experiment
#     report_to='wandb',               # Log metrics to W&B
# )

# # Define custom evaluation function
# @lru_cache()
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     decoded_preds = tokenizer.decode(preds, skip_special_tokens=True)
#     decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)

#     bleu_score = metrics.compute(predictions=[decoded_preds], references=[decoded_labels], rouge_level="1").get("bleu")
#     rouge_score = metrics.compute(predictions=[decoded_preds], references=[decoded_labels]).get("rouge2")
#     meteor_score = metrics.compute(predictions=[decoded_preds], references=[decoded_labels]).get("meteor")

#     return {"bleu": float(bleu_score), "rouge": float(rouge_score), "meteor": float(meteor_score)}

# # Initialize Trainer
# trainer = Seq2SeqTrainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=train_dataset,         # training dataset
#     eval_dataset=eval_dataset,           # validation (and test) dataset
#     compute_metrics=compute_metrics,    # computation of metrics
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
# )

# # Fine-tune the model
# trainer.train()

# # Save the fine-tuned model
# model.save_pretrained('./fine-tuned_codet5')


class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        with open(file_path, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                self.data.append((example["Smelly Sample"].replace("\t"," "), example["Method after Refactoring"].replace("\t"," ") + example["Extracted Method"].replace("\t"," ")))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, target = self.data[idx]
        inputs = self.tokenizer(features, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        labels = self.tokenizer(target, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        # return {"input_ids": inputs.input_ids.squeeze(0), "attention_mask": inputs.attention_mask, "labels": labels.input_ids}
        return {"input_ids": inputs["input_ids"].squeeze(0), "attention_mask": inputs["attention_mask"].squeeze(0), "labels": labels["input_ids"].squeeze(0)}
        # return inputs, labels
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device being used: {device}")

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small").to(device)

train_dataset = CustomDataset("./test-data/train.jsonl", tokenizer)
train_dataloader = DataLoader(train_dataset, 4, True)

val_dataset = CustomDataset("./test-data/val.jsonl", tokenizer)
val_dataloader = DataLoader(val_dataset, 4, True)

# print(next(iter(train_dataloader)))
metric = load("rouge")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    print("Preds:", decoded_preds)
    print("Labels:", decoded_labels)

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,   # batch size per device during training
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy='steps',     # evaluation strategy to use ("no", "steps", "epoch")
    eval_steps=1,                   # Number of update steps between evaluations.
    load_best_model_at_end=True,     # Whether or not to load the best model at the end of training
    metric_for_best_model='bleu',      # metric to use to compare models
    greater_is_better=True,         # whether higher is better for the selected metric
    dataloader_num_workers=2,       # number of workers for loading data
    fp16=False,                      # enable mixed precision training
    run_name='CodeT5-smellfix', # name of this experiment
    predict_with_generate=True,
    generation_max_length=512
    # report_to='wandb',               # Log metrics to W&B
)

trainer = Seq2SeqTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,           # validation (and test) dataset
    compute_metrics=compute_metrics,    # computation of metrics
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    data_collator=None
)

# Fine-tune the model
trainer.train()