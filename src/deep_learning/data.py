from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader, Dataset
import json
from transformers import RobertaTokenizer

class CodeT5Dataset(Dataset):
    def __init__(self, filename, max_length=512):
        self.filename = filename
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.filename, 'r') as f:
            for line in f:
                item = json.loads(line)
                smelly = item['Smelly Sample']
                non_smelly = item['Method after Refactoring']
                extracted = item['Extracted Method']
                data.append((smelly, non_smelly + extracted))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smelly_input, target_output = self.data[idx]

        # Tokenize input and target
        input_ids = self.tokenizer.encode(smelly_input, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        target_ids = self.tokenizer.encode(target_output, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)

        return {'input_ids': input_ids, 'target_ids': target_ids}

if __name__=="__main__":

    dataset = CodeT5Dataset("data/dl/val.jsonl")

    train_data_loader = DataLoader(dataset=dataset)

    for i, data in enumerate(train_data_loader):
        print(i,data)

