"""
Utility functions for TDM distillation: dataset loading, noise scheduling, and diffusion utilities.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


class NoiseScheduler:
    """
    Noise scheduler for diffusion models.
    Handles the forward diffusion process (adding noise) and timestep sampling.
    Uses fp16 for memory efficiency.
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule_type="linear"):
        self.num_train_timesteps = num_train_timesteps

        # Create beta schedule in fp16
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float16)
        elif schedule_type == "scaled_linear":
            # Used in stable diffusion
            self.betas = (torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float16) ** 2)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Precompute useful values in fp16
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float16), self.alphas_cumprod[:-1]])

        # For adding noise
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, original_samples, noise, timesteps):
        """
        Add noise to samples according to the noise schedule.

        Args:
            original_samples: Clean images/latents (x_0)
            noise: Random noise (epsilon ~ N(0, I))
            timesteps: Which timesteps to add noise for

        Returns:
            noisy_samples: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon
        """
        # Convert inputs to fp16
        original_samples = original_samples.half()
        noise = noise.half()

        # Get the appropriate alpha values and move to device
        device = original_samples.device
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(device).flatten()
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(device).flatten()

        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Add noise in fp16
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def sample_timesteps(self, batch_size, device):
        """
        Sample random timesteps for training.

        Args:
            batch_size: Number of timesteps to sample
            device: Device to put timesteps on

        Returns:
            timesteps: Random timesteps in [0, num_train_timesteps)
        """
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()

    def get_velocity(self, sample, noise, timesteps):
        """
        Get velocity prediction target (used in some diffusion models).

        Args:
            sample: Original sample
            noise: Noise added
            timesteps: Current timesteps

        Returns:
            velocity: v = alpha * noise - sigma * sample
        """
        # Convert inputs to fp16
        sample = sample.half()
        noise = noise.half()

        # Get alpha values and move to device
        device = sample.device
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(device).flatten()
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(device).flatten()

        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity


class TDMWeightScheduler:
    """
    Weight scheduler for TDM loss: w(t).
    Higher weights at certain timesteps can improve training.
    """
    def __init__(self, num_timesteps=1000, weight_type="constant", min_weight=1.0, max_weight=5.0):
        self.num_timesteps = num_timesteps
        self.weight_type = weight_type
        self.min_weight = min_weight
        self.max_weight = max_weight

    def get_weight(self, timesteps):
        """
        Get the loss weight for given timesteps.

        Args:
            timesteps: Current timesteps (batch of ints)

        Returns:
            weights: Loss weights for each timestep
        """
        if self.weight_type == "constant":
            return torch.ones_like(timesteps, dtype=torch.float32)

        elif self.weight_type == "snr":
            # Signal-to-noise ratio weighting (higher weight at higher noise levels)
            t_normalized = timesteps.float() / self.num_timesteps
            weights = self.min_weight + (self.max_weight - self.min_weight) * t_normalized
            return weights

        elif self.weight_type == "min_snr":
            # Min-SNR weighting (from "Min-SNR Weighting Strategy" paper)
            t_normalized = timesteps.float() / self.num_timesteps
            snr = (1 - t_normalized) / t_normalized
            weights = torch.clamp(snr, max=5.0)
            return weights

        else:
            raise ValueError(f"Unknown weight type: {self.weight_type}")


def get_dataloader(config):
    """
    Get dataloader for training.
    For diffusion distillation, we typically use image datasets.
    """
    transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    # Replace with actual dataset (e.g., LAION, ImageNet, etc.)
    dataset = datasets.FakeData(
        size=config.get("dataset_size", 10000),
        image_size=(3, config["img_size"], config["img_size"]),
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )


def compute_budget(model):
    """
    Compute model parameter count and FLOPs.
    Assumes fp16 precision.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "memory_mb": total_params * 2 / (1024 ** 2)  # fp16 uses 2 bytes per parameter
    }


def setup_distributed():
    """
    Setup for distributed training (DDP).
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend='nccl')
        return True
    return False
