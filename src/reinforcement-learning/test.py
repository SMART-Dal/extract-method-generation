from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq


# Load datasets
train_dataset = load_dataset("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context", 
                             split="train")
test_dataset = load_dataset("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context", 
                            split="test")
validation_dataset = load_dataset("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-no-context",
                                 split="validation")

print(len(train_dataset[0]["Smelly Sample"]))
print(len(test_dataset))
print(len(validation_dataset))


model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

print(tokenizer.eos_token_id)

def data_preprocessing(examples):
    # example["Smelly Sample"] = example["Smelly Sample"].replace('\n', ' ')
    # example["Method after Refactoring"] = example["Method after Refactoring"].replace('\n', ' ')
    # example["Extracted Method"] = example["Extracted Method"].replace('\n', ' ')
    # print(examples)
    inputs = [example["Smelly Sample"].replace('\n', ' ') for example in examples]
    targets = [example["Extracted Method"].replace('\n', ' ') + " " + example["Method after Refactoring"].replace('\n', ' ') for example in examples]
    model_inputs = tokenizer(inputs, text_pair_target=targets, padding='max_length', truncation=True, max_length=512)
    return model_inputs

train_dataset = train_dataset.map(data_preprocessing, batched=True)
validation_dataset = validation_dataset.map(data_preprocessing, batched=True)

# def data_collator(batch):
#     input_ids = tokenizer(batch["Smelly Sample"], padding='max_length', truncation=True, max_length=512).input_ids
#     targets = tokenizer(batch["Method after Refactoring"]+batch["Extracted Method"],padding="max_length", truncation=True, return_tensors="pt").input_ids

#     return {
#         input_ids: input_ids,
#         targets: targets
#     }

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="Salesforce/codet5-small")


training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=500,
    eval_steps=500,
    save_steps=300
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

trainer.train()