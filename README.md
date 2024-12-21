# Fine-tuning LLMs using PEFT QLoRA for SQL 

## Overview 
This project demonstrates fine-tuning a 11B parameter LLM for generating accurate SQL queries using Parameter-Efficient Fine-Tuning (PEFT) and Quantized Low-Rank Adaptation (QLoRA). By employing 4-bit quantization, it achieves efficient fine-tuning on large language models while maintaining high accuracy with lower computational costs. The Falcon model with quantization achieved around 50% reduction in peak memory usage and 10 tokens/second faster inference speed. 

[View model card of fine-tuned peft adapters via Hugging Face Hub](https://huggingface.co/adityas2410/falcon11b-sql_instruct/tree/main) | [View wandb report](https://api.wandb.ai/links/adityas-ai2410-upwork/58a35uld)

## Key Features
- **Base Model:** Falcon2-11B [tiiuae/falcon-11B](https://huggingface.co/tiiuae/falcon-11B)
- **4-Bit Quantization:** Reduces model memory usage and speeds up fine-tuning while preserving inference accuracy.
- **PEFT with LoRA:** Fine-tunes specific layers of the model, keeping the rest frozen, enabling lightweight training.
- **Instruction Tuning:** Aligns the model to follow prompts effectively for SQL query generation tasks.

## Dependencies
- Python
- PyTorch
- CUDA
- Hugging Face
  - Transformers Provides pre-trained large language models (LLMs), tokenizers, and APIs for model fine-tuning and inference.
  - Datasets: Simplifies dataset loading and preprocessing. Used to fetch and prepare SQL-related datasets for training.
  - Peft: Enables Parameter-Efficient Fine-Tuning (PEFT) to train large models efficiently without modifying all parameters.
  - Accelerate: Optimizes training and inference across multiple GPUs, CPUs, or TPUs, simplifying distributed and mixed-precision training workflows.
  - TRL: Transformer reinforcement learning library used for supervised fine-tuning, specifically for instruction tuning tasks.
- BitsAndBytes: Provides 4-bit and 8-bit quantization techniques, significantly reducing model size and computational requirements for fine-tuning and inference. 
- Weights & Biases (wandb): Tool for tracking, visualizing, and logging metrics during training and evaluation. It enables better monitoring of model performance and hyperparameter optimization.

## Inference
<img width="768" alt="image" src="https://github.com/user-attachments/assets/8208d827-8496-496d-8623-212d7daf8f8e">
<img width="928" alt="image" src="https://github.com/user-attachments/assets/2fdd5425-61ef-4857-92b0-7590d99a9258">

