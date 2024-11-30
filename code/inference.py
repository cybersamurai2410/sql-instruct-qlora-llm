import time 
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

adapters = 'path to peft adapters'
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, # Reduce model weights to 4-bit precision
    bnb_4bit_use_double_quant=True, # Apply additional quantization layer
    bnb_4bit_quant_type="nf4", # Normal float 4-bit format to optimize weights storage
    bnb_4bit_compute_dtype=torch.bfloat16 # 4-bit weights temporarily upscaled to brain float 16-bit for matrix computations
)

# Load local peft adapters with remote base model  
instruction_tuned_model = AutoPeftModelForCausalLM.from_pretrained(
    adapters,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

start_memory = torch.cuda.memory_allocated()
start_time = time.time()

instruction_tuned_model.eval()  # Set model in inference mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = tokenizer(prompt, return_tensors='pt').to(device)

# Use mixed precision and disable gradients
with torch.no_grad(), torch.amp.autocast('cuda'):
    output_tokens = instruction_tuned_model.generate(**batch, max_new_tokens=100)

result = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(result, '\n')

end_time = time.time()
print(f"Inference time: {end_time - start_time} seconds")

end_memory = torch.cuda.memory_allocated()
peak_memory = torch.cuda.max_memory_allocated()
print(f"Memory allocated before inference: {start_memory} bytes")
print(f"Memory allocated after inference: {end_memory} bytes")
print(f"Memory used during inference: {end_memory - start_memory} bytes")
print(f"Peak allocated memory: {peak_memory}")
