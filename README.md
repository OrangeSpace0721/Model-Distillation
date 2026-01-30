# Model Distillation for Flux.2 Generative Image Model

This project provides a codebase for distilling the Flux.2 generative image model into smaller student models, each constrained by a different compute budget, following the methodology described in [Distilling Generative Models with Limited Compute](https://openreview.net/pdf?id=iIGNrDwDuP).

## Features
- Modular code for teacher-student distillation
- Compute budget management for student models
- Distributed training support (PyTorch DDP)
- SLURM job submission scripts for HPC clusters
- Dataset handling and evaluation scripts

## Getting Started
1. Install dependencies (see requirements.txt)
2. Configure your SLURM cluster settings in the provided scripts
3. Run distillation experiments using the main training script

## Project Structure
- `src/` - Core source code (models, training, distillation, utils)
- `scripts/` - SLURM job scripts and experiment launchers
- `configs/` - YAML/JSON config files for experiments
- `requirements.txt` - Python dependencies
- `README.md` - Project overview and instructions

## Reference
- [Distilling Generative Models with Limited Compute (OpenReview)](https://openreview.net/pdf?id=iIGNrDwDuP)

---
*Replace placeholders and configs as needed for your cluster and dataset.*
