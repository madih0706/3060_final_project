# ESE 3060 Final Project Fall 2025

## Project Overview
This project contains two machine learning training benchmarks:
- **airbench94.py**: CIFAR-10 image classification benchmark
- **train_gpt.py**: GPT-2 training on the FineWeb-10B dataset

## Setup and Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (A100/H100 recommended)
- CUDA 11.7 or later

### Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

## Running airbench94.py

### Overview
CIFAR-10 training benchmark achieving 94.01% average accuracy in 3.83 seconds on an NVIDIA A100. You will want to use a single node of an a100.
- CIFAR-10 dataset automatically downloaded on first run
- Cached to `cifar10/` directory as `.pt` files for faster subsequent runs

### Execution
```bash
python airbench94.py
```

Runs 25 training iterations and reports mean/standard deviation accuracy metrics.

### Output
- Per-epoch training metrics (loss, accuracy)
- Validation and test-time augmentation (TTA) accuracy
- Logs saved to `logs/{uuid}/log.pt`

### Hardware Requirements
- NVIDIA A100 GPU recommended
- CUDA 11.7+
- NVIDIA Driver 515.105.01 or compatible

### Reference
Based on: [cifar10-airbench legacy airbench94.py](https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench94.py)

## Running train_gpt.py

### Overview
Trains a GPT-2 model on the FineWeb-10B dataset. You will want to use an 8xH100.

### Execution
Download the data with 
```bash
python cached_fineweb10B.py 9
```
and then run the script with 
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Hardware Requirements
- Tested on 8× NVIDIA H100 80GB GPUs
- PyTorch 2.4.1+ with CUDA 12.1

### Reference
Based on: [modded-nanogpt record number #5](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_1_short/2024-10-14_ModernArch/dabaaddd-237c-4ec9-939d-6608a9ed5e27.txt)