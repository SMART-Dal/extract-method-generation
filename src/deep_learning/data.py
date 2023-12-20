from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
import json

class CodeT5Dataset(IterableDataset):
    def __init__(self, filename):
        self.filename = filename

    def parser(self):
        with open(self.filename, 'r') as f:
            for line in f:
                item = json.loads(line)
                smelly = item['Smelly Sample']
                non_smelly = item['Method after Refactoring']
                extracted = item['Extracted Method']
                yield smelly, non_smelly + extracted

    def __iter__(self):
        return self.parser()
    

dataset = CodeT5Dataset("data/dl/val.jsonl")

train_data_loader = DataLoader(dataset=dataset)

for i, data in enumerate(train_data_loader):
    print(i,data)

