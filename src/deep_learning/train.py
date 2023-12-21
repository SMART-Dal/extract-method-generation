import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from .data import CodeT5Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small').to(device)

# Fine-tuning parameters
epochs = 5
learning_rate = 1e-4
batch_size = 16

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

train_dataset = CodeT5Dataset("data/dl/train.jsonl")
train_data = DataLoader(dataset=train_dataset, batch_size=batch_size)


val_dataset = CodeT5Dataset("data/dl/val.jsonl")
val_data = DataLoader(dataset=val_dataset, batch_size=batch_size)


for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_data):
        optimizer.zero_grad()
        
        # Tokenize input and target sequences
        # input_ids = tokenizer.encode(input_seq, return_tensors='pt', max_length=512, truncation=True).to(device)
        # target_ids = tokenizer.encode(target_seq, return_tensors='pt', max_length=512, truncation=True).to(device)

        input_ids = batch["input_ids"].squeeze(0).to(device)
        target_ids = batch["target_ids"].squeeze(0).to(device)

        print(input_ids)
        print(target_ids)

        # Forward pass
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        total_loss = 0
        for input_seq_val, target_seq_val in tqdm(val_data):
            # Tokenize and move to GPU
            # input_ids_val = tokenizer.encode(input_seq_val, return_tensors='pt', max_length=512, truncation=True).to(device)
            # target_ids_val = tokenizer.encode(target_seq_val, return_tensors='pt', max_length=512, truncation=True).to(device)

            input_ids_val = batch["input_ids"].to(device)
            target_ids_val = batch["target_ids"].to(device)

            # Forward pass for validation
            outputs_val = model(input_ids_val, labels=target_ids_val)
            loss_val = outputs_val.loss
            total_loss += loss_val.item()

        average_loss = total_loss / len(val_data)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {average_loss}")
