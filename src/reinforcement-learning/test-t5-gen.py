from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    DefaultDataCollator,
    T5ForConditionalGeneration
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead, create_reference_model, set_seed
from trl.core import LengthSampler, respond_to_batch

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

rl_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained("Salesforce/codet5-small")
t5model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")


text = "def greet(user): print(f'hello <extra_id_0>!')"
input_ids = tokenizer(text, return_tensors="pt").input_ids

print("=======Auto Model===========")
print(tokenizer.decode(model.generate(input_ids)[0],skip_special_tokens=True))
print("=======T5 Model===========")
print(tokenizer.decode(t5model.generate(input_ids)[0],skip_special_tokens=True))
print("=======RL Model===========")
print(tokenizer.decode(rl_model.generate(input_ids)[0],skip_special_tokens=True))

