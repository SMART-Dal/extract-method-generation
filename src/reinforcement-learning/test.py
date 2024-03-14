import argparse
import multiprocessing
import os
from utils import get_filenames, read_refactoring_examples, convert_examples_to_features, load_data
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

parser = argparse.ArgumentParser()

parser.add_argument("--train_filename", default=None, type=str,
                    help="The train filename. Should contain the .jsonl files for this task.")
parser.add_argument("--dev_filename", default=None, type=str,
                    help="The dev filename. Should contain the .jsonl files for this task.")
parser.add_argument("--test_filename", default=None, type=str,
                    help="The test filename. Should contain the .jsonl files for this task.")

args = parser.parse_args()

print(args.dev_filename)

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")

# pool = multiprocessing.Pool(os.cpu_count())
args.train_filename, args.dev_filename, args.test_filename = get_filenames("src/refactoring-fine-tuning/data", "refactoring", None, None, "no-context")
print(args.dev_filename)
dev_examples, dev_data = load_data(args, args.dev_filename, tokenizer, 'dev')
dev_sampler = RandomSampler(dev_data)
dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=32)

