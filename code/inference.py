import torch
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, TrainerCallback
from peft import AutoPeftModelForCausalLM, PeftModel, PeftConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

model_id = "adityas2410/falcon11b-sql_instruct"
instruction_tuned_model = AutoPeftModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    trust_remote_code=True,
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Give three benifits of fine-tuning language models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = tokenizer(prompt, return_tensors='pt').to(device)

# Auotmatically handle mixed precision during text generation
with torch.cuda.amp.autocast():
  output_tokens = peft_model.generate(**batch, max_new_tokens=100)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
