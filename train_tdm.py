"""
Example training script for TDM (Trajectory Distribution Matching) distillation.

This script demonstrates how to:
1. Set up the TDM distiller with proper config
2. Load data
3. Run the training loop
4. Save checkpoints
"""

import torch
from src.distill import TDMDistiller
from src.utils import get_dataloader
import os


def main():
    # ========== Configuration ==========
    config = {
        # Model paths
        "flux_teacher_path": "./flux2_teacher",  # Path to pretrained Flux model

        # Training hyperparameters
        "epochs": 100,
        "batch_size": 4,  # Adjust based on GPU memory (fp16 allows larger batches!)
        "lr_student": 1e-5,
        "lr_estimator": 1e-5,
        "weight_decay": 0.01,

        # TDM-specific parameters
        "student_steps": 4,  # Number of inference steps for student (1-4)
        "num_train_timesteps": 1000,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "schedule_type": "linear",

        # Loss weighting
        "weight_type": "snr",  # Options: "constant", "snr", "min_snr"
        "min_weight": 1.0,
        "max_weight": 5.0,

        # Training stability
        "gradient_accumulation_steps": 1,
        "update_estimator_every": 1,  # Update estimator every N steps
        "use_student_generation": True,  # Use student-generated samples

        # Mixed precision (fp16)
        "use_amp": True,  # Use automatic mixed precision with GradScaler

        # Data
        "img_size": 512,  # Image size (512 for SDXL, 1024 for Flux)
        "dataset_size": 10000,
        "num_workers": 4,

        # Device and checkpointing
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "checkpoint_dir": "./checkpoints",
        "save_every": 10,  # Save checkpoint every N epochs
        "steps_per_epoch": None,  # Will be set based on dataloader
    }

    # Create checkpoint directory
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    print("=" * 60)
    print("TDM (Trajectory Distribution Matching) Distillation")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Precision: fp16 (float16)")
    print(f"Mixed Precision (AMP): {config['use_amp']}")
    print(f"Student inference steps: {config['student_steps']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate (Student): {config['lr_student']}")
    print(f"Learning rate (Estimator): {config['lr_estimator']}")
    print("=" * 60)

    # ========== Load Data ==========
    print("\nLoading dataset...")
    dataloader = get_dataloader(config)
    config["steps_per_epoch"] = len(dataloader)
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # ========== Initialize TDM Distiller ==========
    print("\nInitializing TDM Distiller...")
    distiller = TDMDistiller(config)

    # Optional: Load from checkpoint
    checkpoint_path = config.get("resume_from", None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        distiller.load_checkpoint(checkpoint_path)

    # ========== Train ==========
    print("\nStarting training...\n")
    distiller.train(dataloader)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
