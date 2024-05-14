import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq

def save_weights(model_name, tokenizer_name):
    # if not output_path:
    #     output_path = os.makedirs(os.path.join(os.path.dirname(__file__),"pre_trained_weights"),exist_ok=True)

       
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model_output_path = os.path.join(os.path.dirname(__file__),"pre_trained_weights",model_name.replace('/','-'))
    os.makedirs(model_output_path, exist_ok=True)
    model.save_pretrained(model_output_path, from_pt=True)

    tokenizer_save_path = os.path.join(os.path.dirname(__file__),"pre_trained_weights",model_name.replace('/','-')+"-tokenizer")
    os.makedirs(tokenizer_save_path)
    tokenizer.save_pretrained(tokenizer_save_path,exist_ok=True)



if __name__=="__main__":
    save_weights("codellama/CodeLlama-7b-hf","codellama/CodeLlama-7b-hf")