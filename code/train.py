import torch
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, TrainerCallback
from peft import AutoPeftModelForCausalLM, PeftModel, PeftConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# https://huggingface.co/tiiuae/tiiuae/falcon-11B
model_id = "tiiuae/falcon-11B"
device = 0 if torch.cuda.is_available() else -1

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, # Reduce model weights to 4-bit precision
    bnb_4bit_use_double_quant=True, # Apply additional quantization layer
    bnb_4bit_quant_type="nf4", # Normal float 4-bit format to optimize weights storage
    bnb_4bit_compute_dtype=torch.bfloat16 # 4-bit weights temporarily upscaled to brain float 16-bit for matrix computations
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
    )
model.config.pretraining_tp = 1 # Tensor parallelism for distributed computing; 1 degree operates on single GPU

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Set padding token same as end-of-sequence token
tokenizer.padding_side = "right"

model.gradient_checkpointing_enable() # Gradient checkpointing for memory efficiency
model = prepare_model_for_kbit_training(model) # Freezes layers for quantization except for those specified for fine-tuning via peft lora config

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

data = load_dataset("HuggingFaceH4/testing_self_instruct_small", split="train")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"### Question: {example['prompt'][i]}\n ### Answer: {example['completion'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

path = "/content/drive/MyDrive/falcon11b-sql_instruct"

training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir='falcon-mamba_instruct'
        )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_datasets['train'],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False) # Ensures batched data is padded correctly using information from the tokenizer
    )

model.config.use_cache = False
trainer.train()
trainer.save_model()

# Supervised Fine-Tuning
sft_config = SFTConfig(
    output_dir=path, # Directory to save the fine-tuned model (mount drive)
    overwrite_output_dir=True, # Overwrites the output directory if it exists
    num_train_epochs=3, # number of examples / batch size per step = total steps per epoch
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    weight_decay=0.001, # Regularization
    max_grad_norm=0.3, # Gradient clipping
    warmup_ratio=0.03,  # n*100% steps before scheduler
    optim="paged_adamw_32bit",
    per_device_train_batch_size=8, # Batch size per device (GPU); examples processed per step
    gradient_accumulation_steps=4, # Number of steps before applying gradients (total accumulated gradients applied on nth step; delaying parameter update to handle large batch sizes e.g. 4 steps accumulated * 8 batches = 32 examples then update params)
    gradient_checkpointing=True, # Save memory by recomputing activations instead of storing in memory
    save_steps=100, # Saves the model state every n steps
    logging_dir=f"{path}/logs",
    logging_steps=25, # Log training metrics every n steps
    max_seq_length=2048, # Set to max context length of llm
    packing=True, # Combines sequences to fit context length (not compatilble with DataCollatorForCompletionOnlyLM)
    report_to="wandb", # Logging to Weights & Biases
)

sfttrainer = SFTTrainer(
    model,
    train_dataset=data,
    args=sft_config,
    peft_config=lora_config,
    formatting_func=formatting_prompts_func, # The `formatting_func` should return a list of processed strings since it can lead to silent bugs.
    data_collator=collator,
    tokenizer=tokenizer,
    )

sfttrainer.train()
sfttrainer.save_model()
wandb.finish()

from huggingface_hub import notebook_login

notebook_login()
repo_id = "adityas2410/falcon11b-sql_instruct"
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
