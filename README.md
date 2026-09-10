# Robust Online Federated Learning: Dealing with Switching Byzantine Clients

This repository contains the implementation and experimental framework for the paper *Robust Online Federated Learning: Dealing with Switching Byzantine Clients*. It provides tools for evaluating robust federated learning algorithms in the presence of switching Byzantine clients.

## Project Structure

```
RobustOnlineFedLearning/
├── benchmark/                # Benchmark framework adapted from ByzFL
│   ├── benchmark.py
│   ├── evaluate_results.py
│   ├── managers.py
│   └── train.py
├── config/                   # Configuration files
│   ├── config_cifar.json
│   └── config_mnist.json
├── data/                     # Downloaded datasets
│   ├── MNIST/
│   └── CIFAR10/
├── models/                   # Saved model checkpoints
├── plot/                     # Generated figures
├── results/                  # Experiment results
├── src/
│   ├── aggregation_time.py   # Aggregation time generation
│   └── clients.py            # Modifications to the ByzFL client implementation
├── pyproject.toml
├── uv.lock
├── LICENSE.txt
├── README.md
└── main.py                   # Entry point for training and plotting
```

## Installation

### Requirements

- Python >= 3.13
- CUDA-compatible GPU (optional, for GPU execution)
- [uv](https://docs.astral.sh/uv/)

### Setup

```bash
uv sync
```

## Configuration

Experiments are defined using JSON configuration files in `config/`.
The configuration specifies, among other parameters:

- Training algorithm and hyperparameters
- Number of training steps
- Number of clients and Byzantine clients
- Data distribution
- Dataset and model
- (Pre-)aggregation rule
- Byzantine attack
- Evaluation and output settings

See `config/config_mnist.json` and `config/config_cifar.json` for examples.

## Running Experiments

### Basic Usage

Run experiments using a JSON configuration file:

```bash
uv run main.py learning --config config/config_mnist.json
```

### Parallel Execution

Use `--n-jobs` to specify the number of independent training experiments to run concurrently:

```bash
# Run 4 experiments in parallel
uv run main.py learning \
    --config config/config_mnist.json \
    --n-jobs 4
```

When running many experiments concurrently, limiting the number of CPU threads per worker may be necessary to avoid thread oversubscription or system thread limits. For example:

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
uv run main.py learning \
    --config config/config_mnist.json \
    --n-jobs 4
```

### GPU Support

Use `--gpus` to select the physical GPUs on which experiments are allowed to run:

```bash
# Run experiments on GPUs 0, 1, and 3
uv run main.py learning \
    --config config/config_mnist.json \
    --n-jobs 3 \
    --gpus 0 1 3
```

Jobs are distributed across the selected GPUs in round-robin order. The number of parallel jobs is independent of the number of GPUs, so multiple experiments can run concurrently on each GPU:

```bash
# Run 2 experiments per GPU
uv run main.py learning \
    --config config/config_mnist.json \
    --n-jobs 6 \
    --gpus 0 1 3
```

Each worker is restricted to its assigned physical GPU using `CUDA_VISIBLE_DEVICES`.

If `--gpus` is omitted, GPU visibility is left unchanged. In particular, when multiple GPUs are visible and the configuration uses `"device": "cuda"`, models may use PyTorch `DataParallel` across the visible GPUs.

## Plotting Results

After running experiments, generate plots:

```bash
uv run main.py plot --dataset mnist
```

## Citation

If you use this repository, please cite our work:

```latex
@misc{godichonbaggioni2026robustonlinefl,
  title  = {Robust Online Federated Learning: Dealing with Switching Byzantine Clients},
  author = {Antoine Godichon-Baggioni and Rafael Pinot and Pierre Tarrago},
  year   = {2026},
}
```

This implementation builds on ByzFL. Please also cite:

```latex
@misc{gonzález2025byzflresearchframeworkrobust,
  title     = {ByzFL: Research Framework for Robust Federated Learning},
  author    = {Marc González and Rachid Guerraoui and Rafael Pinot and Geovani Rizk and John Stephan and François Taïani},
  year      = {2025},
  eprint    = {2505.24802},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2505.24802}
}
```

## Acknowledgements

Parts of the experimental framework and of the library are adapted from ByzFL, Copyright (c) 2024 EPFL, and licensed under the MIT License. The original license and copyright notice are retained in `LICENSE.txt`.
