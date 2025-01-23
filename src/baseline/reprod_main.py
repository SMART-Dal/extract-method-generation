import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader, Dataset
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from javalang.tokenizer import tokenize

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, dropout_rate):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.bilstm1 = nn.LSTM(embed_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output1, _ = self.bilstm1(embedded)
        output2, _ = self.bilstm2(output1)
        return self.dropout(output2)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, timestep, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, dropout_rate):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(hidden_dim * 2 + embed_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 3, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, _) = self.lstm(lstm_input)
        output = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        prediction = self.fc(output)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.size(1)
        batch_size = trg.size(0)
        trg_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs = self.encoder(src)
        hidden = encoder_outputs[:, -1, :]
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t, :] = output
            top1 = output.argmax(1)
            input_token = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

class CodeDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append((tokenizer(item['Input'])[0], tokenizer(item['Output'])[0]))
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, trg = self.data[idx]
        src = src[:self.max_len] + [0] * (self.max_len - len(src))
        trg = trg[:self.max_len] + [0] * (self.max_len - len(trg))
        return torch.tensor(src), torch.tensor(trg)

# Tokenization for Java code
def tokenize_java_code(java_code):
    token_list = []
    token_map = {}
    identifier_index = 1

    for token in tokenize(java_code):
        token_type = type(token).__name__
        if token_type in ['Identifier', 'DecimalInteger']:
            replacement = f"{token_type.upper()}_{identifier_index}"
            token_map[replacement] = token.value
            token_list.append(replacement)
            identifier_index += 1
        elif token_type in ['Boolean', 'Integer']:
            token_list.append(token.value)
        else:
            token_list.append(token.value)

    return token_list, token_map

def decode_tokens(token_list, token_map):
    return ' '.join(token_map.get(tok, tok) for tok in token_list)

# Training loop
def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg)
            output = output[:, 1:].reshape(-1, OUTPUT_DIM)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

# Evaluation function
def evaluate_model(model, test_loader, tokenizer, device):
    model.eval()
    references, hypotheses = [], []
    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            output = output.argmax(2).cpu().numpy()
            for i in range(len(trg)):
                references.append([decode_tokens(trg[i].cpu().numpy(), tokenizer)])
                hypotheses.append(decode_tokens(output[i], tokenizer))
    bleu = corpus_bleu(references, hypotheses)
    meteor = sum(meteor_score(ref, hyp) for ref, hyp in zip(references, hypotheses)) / len(hypotheses)
    print(f"BLEU Score: {bleu:.4f}, METEOR Score: {meteor:.4f}")

# Hyperparameters
INPUT_DIM = 10000
OUTPUT_DIM = 10000
EMBED_DIM = 64
ENC_HIDDEN_DIM = 64
DEC_HIDDEN_DIM = 256
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10
MAX_LEN = 50

# Instantiate model
encoder = Encoder(INPUT_DIM, EMBED_DIM, ENC_HIDDEN_DIM, DROPOUT_RATE)
decoder = Decoder(OUTPUT_DIM, EMBED_DIM, DEC_HIDDEN_DIM, DROPOUT_RATE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(encoder, decoder, device).to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Datasets and DataLoaders
tokenizer = tokenize_java_code
train_dataset = CodeDataset('train.jsonl', tokenizer, MAX_LEN)
val_dataset = CodeDataset('val.jsonl', tokenizer, MAX_LEN)
test_dataset = CodeDataset('test.jsonl', tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Train and evaluate
train_model(model, train_loader, val_loader, optimizer, criterion, device, NUM_EPOCHS)
# evaluate_model(model, test_loader, tokenizer, device)
