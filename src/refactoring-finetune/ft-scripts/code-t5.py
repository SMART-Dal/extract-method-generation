import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

train_dataset = load_dataset(data_files="/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context/train.jsonl",
                    split="train")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")

def preprocess_function(examples):
    inputs = [ex["Smelly Sample"] for ex in examples]
    targets = [ex["Method after Refactoring"] for ex in examples]

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
    # num_proc=data_args.preprocessing_num_workers,
    # remove_columns=column_names,
    # load_from_cache_file=not data_args.overwrite_cache,
    desc="Running tokenizer on train dataset",
)

metric = evaluate.load("sacrebleu", cache_dir="./cache_dir")

# Data collator
# label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
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