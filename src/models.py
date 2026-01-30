"""
Model definitions for TDM (Trajectory Distribution Matching) distillation.
Includes Teacher (frozen), Student (trainable), and Score Estimator (trainable).
"""

import torch
import torch.nn as nn
from diffusers import FluxPipeline
from copy import deepcopy


class DiffusionModelWrapper(nn.Module):
    """
    Wrapper for diffusion models that provides score (noise) prediction.
    Works with Flux and other diffusion transformers.
    """
    def __init__(self, transformer_model):
        super().__init__()
        self.transformer = transformer_model

    def predict_noise(self, noisy_latents, timesteps, encoder_hidden_states=None, pooled_projections=None):
        """
        Predict noise (score) given noisy latents and timesteps.
        This is the core function for score matching.

        Args:
            noisy_latents: Noisy image latents (z_t)
            timesteps: Diffusion timesteps (t)
            encoder_hidden_states: Text embeddings (for conditional generation)
            pooled_projections: Pooled text embeddings

        Returns:
            predicted_noise: The predicted noise (epsilon)
        """
        # Flux models output noise predictions directly
        model_output = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            return_dict=False
        )[0]
        return model_output

    def forward(self, noisy_latents, timesteps, encoder_hidden_states=None, pooled_projections=None):
        """Forward pass for compatibility."""
        return self.predict_noise(noisy_latents, timesteps, encoder_hidden_states, pooled_projections)


class Flux2Teacher(nn.Module):
    """
    Teacher model (frozen) - provides the 'Real Score' for TDM.
    This is the original pretrained Flux model.
    Uses fp16 for memory efficiency.
    """
    def __init__(self, config):
        super().__init__()
        model_path = config.get("flux_teacher_path", "./flux2_teacher")

        # Load the full pipeline in fp16
        self.pipeline = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16"
        )

        # Extract the transformer (the actual diffusion model)
        self.transformer = self.pipeline.transformer

        # Wrap in our diffusion wrapper
        self.model = DiffusionModelWrapper(self.transformer)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Ensure model is in fp16
        self.half()

    def predict_noise(self, noisy_latents, timesteps, encoder_hidden_states=None, pooled_projections=None):
        """Compute the Real Score (ground truth gradient towards valid data)."""
        # Ensure inputs are fp16
        noisy_latents = noisy_latents.half()
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.half()
        if pooled_projections is not None:
            pooled_projections = pooled_projections.half()

        with torch.no_grad():
            return self.model.predict_noise(noisy_latents, timesteps, encoder_hidden_states, pooled_projections)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states=None, pooled_projections=None):
        return self.predict_noise(noisy_latents, timesteps, encoder_hidden_states, pooled_projections)


class StudentModel(nn.Module):
    """
    Student model (trainable) - learns to match the teacher's trajectory in 1-4 steps.
    Initialized as a copy of the Teacher.
    Uses fp16 for memory efficiency.
    """
    def __init__(self, teacher_model, config):
        super().__init__()
        # Create a deep copy of the teacher's transformer
        self.transformer = deepcopy(teacher_model.transformer)

        # Wrap in our diffusion wrapper
        self.model = DiffusionModelWrapper(self.transformer)

        # Student is trainable
        for param in self.parameters():
            param.requires_grad = True

        self.num_inference_steps = config.get("student_steps", 4)

        # Convert to fp16
        self.half()

    def predict_noise(self, noisy_latents, timesteps, encoder_hidden_states=None, pooled_projections=None):
        """Predict noise to denoise the input."""
        # Ensure inputs are fp16
        noisy_latents = noisy_latents.half()
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.half()
        if pooled_projections is not None:
            pooled_projections = pooled_projections.half()

        return self.model.predict_noise(noisy_latents, timesteps, encoder_hidden_states, pooled_projections)

    def generate(self, noise, timesteps, encoder_hidden_states=None, pooled_projections=None):
        """
        Generate image from noise using few-step inference.
        This is used during training to create 'fake' samples.
        """
        latents = noise.half()
        for t in timesteps[:self.num_inference_steps]:
            # Predict noise
            noise_pred = self.predict_noise(latents, t, encoder_hidden_states, pooled_projections)

            # Simple Euler step (can be replaced with better solvers)
            alpha = torch.tensor(1.0 - t.float() / 1000.0, dtype=torch.float16, device=latents.device)
            latents = latents - alpha * noise_pred

        return latents

    def forward(self, noisy_latents, timesteps, encoder_hidden_states=None, pooled_projections=None):
        return self.predict_noise(noisy_latents, timesteps, encoder_hidden_states, pooled_projections)


class ScoreEstimator(nn.Module):
    """
    Score Estimator (trainable) - acts as adversary, learns to predict the Student's CURRENT distribution.
    Provides the 'Fake Score' for TDM.
    Initialized as a copy of the Teacher.
    Uses fp16 for memory efficiency.
    """
    def __init__(self, teacher_model):
        super().__init__()
        # Create a deep copy of the teacher's transformer
        self.transformer = deepcopy(teacher_model.transformer)

        # Wrap in our diffusion wrapper
        self.model = DiffusionModelWrapper(self.transformer)

        # Estimator is trainable
        for param in self.parameters():
            param.requires_grad = True

        # Convert to fp16
        self.half()

    def predict_noise(self, noisy_latents, timesteps, encoder_hidden_states=None, pooled_projections=None):
        """Compute the Fake Score (gradient towards Student's current distribution)."""
        # Ensure inputs are fp16
        noisy_latents = noisy_latents.half()
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.half()
        if pooled_projections is not None:
            pooled_projections = pooled_projections.half()

        return self.model.predict_noise(noisy_latents, timesteps, encoder_hidden_states, pooled_projections)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states=None, pooled_projections=None):
        return self.predict_noise(noisy_latents, timesteps, encoder_hidden_states, pooled_projections)
