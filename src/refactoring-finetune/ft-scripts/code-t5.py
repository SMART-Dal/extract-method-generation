import torch
import evaluate
import numpy as np
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    HfArgumentParser
    )

from dataclasses import dataclass, field
from typing import Optional

os.environ["WANDB_PROJECT"]="extract_method_refactoring_generation"

@dataclass
class ScriptArguments:
    """
    The config class
    """

    model_name: Optional[str] = field(default="Salesforce/codet5-small", metadata={"help": "the model name or path"})
    tokenizer_name: Optional[str] = field(default="Salesforce/codet5-small", metadata={"help": "the model name or path"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default="./output/codet5-out",
        metadata={"help": "the path to save the model"},
    )
    run_name: Optional[str] = field(default=None, metadata={"help":"the name of the experiment"})
    train_data_file_path: Optional[str] = field(default=None,metadata={"help":"the path to the train data file"})
    eval_data_file_path: Optional[str] = field(default=None,metadata={"help":"the path to the eval data file"})
    test_data_file_path: Optional[str] = field(default=None,metadata={"help":"the path to the eval data file"})
    num_epochs: Optional[int] = field(default=3,metadata={"help":"number of training epochs"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


train_dataset = load_dataset("json",data_files=script_args.train_data_file_path, split="train")

eval_dataset = load_dataset("json",data_files=script_args.eval_data_file_path, split="train")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_name)

def preprocess_function(examples):
    # for e in examples:
    #     print(e)
    #     break
    # inputs = [ex["Smelly Sample"] for ex in examples]
    # targets = [ex["Method after Refactoring"] for ex in examples]
    # inputs = examples["Smelly Sample"]
    # targets = examples["Method after Refactoring"]
    inputs = examples["Input"]
    targets = examples["Output"]    

    padding = "max_length"
    # inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=512, padding=padding, truncation=True)
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=512, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    # if padding == "max_length" and data_args.ignore_pad_token_for_loss:
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    desc="Running tokenizer on train dataset",
)

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    desc="Running tokenizer on eval dataset"
)

metric1 = evaluate.load("sacrebleu", cache_dir="./cache_dir")
metric2 = evaluate.load("rouge")

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    # pad_to_multiple_of=8 if training_args.fp16 else None,
)
# if data_args.pad_to_max_length:
#     data_collator = default_data_collator
# else:
#     data_collator = DataCollatorForSeq2Seq(
#         tokenizer,
#         model=model,
#         label_pad_token_id=label_pad_token_id,
#         # pad_to_multiple_of=8 if training_args.fp16 else None,
#     )

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result_bleu = metric1.compute(predictions=decoded_preds, references=decoded_labels)
    result_bleu = {"bleu": result_bleu["score"]}

    result_rouge = metric2.compute(predictions=decoded_preds, references=decoded_labels)

    result = {**result_bleu, **result_rouge}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

training_args = Seq2SeqTrainingArguments(
    run_name=script_args.run_name,
    output_dir=script_args.model_save_path,
    overwrite_output_dir=True,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    num_train_epochs=script_args.num_epochs,
    generation_max_length=512,
    logging_steps=1
)

trainer = Seq2SeqTrainer(
    model=model,
    # device='cuda',
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

train_result = trainer.train()

trainer.save_model()

metrics = train_result.metrics
# max_train_samples = (
#     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
# )
# metrics["train_samples"] = min(max_train_samples, len(train_dataset))

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# python code-t5.py --model_save_path ./output/codet5-test --run_name code_t5_test --train_data_file_path /home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract
# -method-generation/data/dl-no-context-len/train.jsonl --eval_data_file_path /home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context-len/val.jsonl --num_epochs 
# 1