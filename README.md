# Fine-tuning LLMs using PEFT QLoRA for SQL 

## Overview 
This project demonstrates fine-tuning a 11B parameter LLM for generating accurate SQL queries using Parameter-Efficient Fine-Tuning (PEFT) and Quantized Low-Rank Adaptation (QLoRA). By employing 4-bit quantization, it achieves efficient fine-tuning on large language models while maintaining high accuracy with lower computational costs.

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

