"""
TDM (Trajectory Distribution Matching) training loop for diffusion distillation.

This implements the state-of-the-art method for distilling diffusion models into fast 1-4 step generators.
Key components:
- Teacher (frozen): Provides Real Score (ground truth)
- Student (trainable): Learns fast inference
- Estimator (trainable): Acts as adversary, learns Student's current distribution
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from .models import Flux2Teacher, StudentModel, ScoreEstimator
from .utils import NoiseScheduler, TDMWeightScheduler, compute_budget


class TDMDistiller:
    """
    Trajectory Distribution Matching (TDM) Distiller.

    Implements dual-model training loop where Student learns to match Teacher's trajectory
    while Estimator acts as adversary to prevent distribution collapse.
    """

    def __init__(self, config):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Step A: Initialize the three models in fp16
        print("Initializing TDM models (fp16)...")
        self.teacher = Flux2Teacher(config).to(self.device)
        self.student = StudentModel(self.teacher, config).to(self.device)
        self.estimator = ScoreEstimator(self.teacher).to(self.device)

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Initialize noise scheduler and weight scheduler
        self.noise_scheduler = NoiseScheduler(
            num_train_timesteps=config.get("num_train_timesteps", 1000),
            beta_start=config.get("beta_start", 0.0001),
            beta_end=config.get("beta_end", 0.02),
            schedule_type=config.get("schedule_type", "linear")
        )

        self.weight_scheduler = TDMWeightScheduler(
            num_timesteps=config.get("num_train_timesteps", 1000),
            weight_type=config.get("weight_type", "snr"),
            min_weight=config.get("min_weight", 1.0),
            max_weight=config.get("max_weight", 5.0)
        )

        # Step A: Initialize separate optimizers for Student and Estimator
        self.optimizer_student = optim.AdamW(
            self.student.parameters(),
            lr=config.get("lr_student", 1e-5),
            betas=(0.9, 0.999),
            weight_decay=config.get("weight_decay", 0.01)
        )

        self.optimizer_estimator = optim.AdamW(
            self.estimator.parameters(),
            lr=config.get("lr_estimator", 1e-5),
            betas=(0.9, 0.999),
            weight_decay=config.get("weight_decay", 0.01)
        )

        # Learning rate schedulers (optional but recommended)
        self.scheduler_student = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_student,
            T_max=config.get("epochs", 100) * config.get("steps_per_epoch", 1000)
        )

        self.scheduler_estimator = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_estimator,
            T_max=config.get("epochs", 100) * config.get("steps_per_epoch", 1000)
        )

        # Training hyperparameters
        self.update_estimator_every = config.get("update_estimator_every", 1)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

        # Mixed precision training with GradScaler for fp16
        self.use_amp = config.get("use_amp", True) and self.device == "cuda"
        if self.use_amp:
            self.scaler_student = torch.cuda.amp.GradScaler()
            self.scaler_estimator = torch.cuda.amp.GradScaler()
            print("Using automatic mixed precision (AMP) with fp16")
        else:
            self.scaler_student = None
            self.scaler_estimator = None

        # Print model info
        student_budget = compute_budget(self.student)
        estimator_budget = compute_budget(self.estimator)
        print(f"Student: {student_budget['trainable_params']:,} trainable params ({student_budget['memory_mb']:.2f} MB)")
        print(f"Estimator: {estimator_budget['trainable_params']:,} trainable params ({estimator_budget['memory_mb']:.2f} MB)")

    def compute_tdm_loss(self, latents, timesteps, encoder_hidden_states=None, pooled_projections=None):
        """
        Compute the TDM loss: L_TDM = w(t) * ||epsilon_real - epsilon_fake||^2

        Args:
            latents: Noisy latents (x_t)
            timesteps: Current timesteps (t)
            encoder_hidden_states: Text embeddings (optional)
            pooled_projections: Pooled text embeddings (optional)

        Returns:
            loss: TDM loss
            real_score: Teacher's noise prediction
            fake_score: Estimator's noise prediction
        """
        # Step C: Compute Real Score (from frozen Teacher)
        with torch.no_grad():
            epsilon_real = self.teacher.predict_noise(
                latents, timesteps, encoder_hidden_states, pooled_projections
            )

        # Step D: Compute Fake Score (from trainable Estimator)
        epsilon_fake = self.estimator.predict_noise(
            latents, timesteps, encoder_hidden_states, pooled_projections
        )

        # Step E: Compute TDM loss with weighting
        weights = self.weight_scheduler.get_weight(timesteps).to(self.device)

        # Expand weights to match score dimensions
        while len(weights.shape) < len(epsilon_real.shape):
            weights = weights.unsqueeze(-1)

        # TDM loss: weighted difference between real and fake scores
        score_difference = epsilon_real - epsilon_fake
        loss = weights * (score_difference ** 2)
        loss = loss.mean()

        return loss, epsilon_real, epsilon_fake

    def train_step(self, batch, step):
        """
        Single training step of TDM with fp16 mixed precision.

        This implements the core TDM algorithm:
        1. Generate fake samples with Student
        2. Add noise to get noisy latents
        3. Compute Real Score (Teacher) and Fake Score (Estimator)
        4. Update Student to minimize TDM loss
        5. Update Estimator to track Student's distribution
        """
        # Step B: Get data and sample noise
        # Convert images to fp16 immediately
        images = batch[0].to(self.device).half()
        batch_size = images.shape[0]

        # Sample random timesteps
        timesteps = self.noise_scheduler.sample_timesteps(batch_size, self.device)

        # Step B: Sample random noise in fp16
        noise = torch.randn_like(images, dtype=torch.float16)

        # Generate latents: either from Student's generation or by adding noise to images
        # For TDM, we typically generate from Student to create "fake" samples
        if self.config.get("use_student_generation", True) and torch.rand(1).item() > 0.5:
            # Generate from Student (creates distribution mismatch)
            with torch.no_grad():
                student_generated = self.student.generate(
                    noise, timesteps, None, None
                )
            # Add noise to student's output
            noisy_latents = self.noise_scheduler.add_noise(student_generated, noise, timesteps)
        else:
            # Add noise directly to real images (standard diffusion training)
            noisy_latents = self.noise_scheduler.add_noise(images, noise, timesteps)

        # ===== Update Student =====
        self.student.train()
        self.estimator.eval()

        # Compute TDM loss with autocast for mixed precision
        if self.use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                tdm_loss, epsilon_real, epsilon_fake = self.compute_tdm_loss(
                    noisy_latents, timesteps, None, None
                )
                tdm_loss_student = tdm_loss / self.gradient_accumulation_steps

            # Backprop for Student with gradient scaling
            self.scaler_student.scale(tdm_loss_student).backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping (important for stability)
                self.scaler_student.unscale_(self.optimizer_student)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)

                self.scaler_student.step(self.optimizer_student)
                self.scaler_student.update()
                self.optimizer_student.zero_grad()
                self.scheduler_student.step()
        else:
            # Without AMP, regular fp16 computation
            tdm_loss, epsilon_real, epsilon_fake = self.compute_tdm_loss(
                noisy_latents, timesteps, None, None
            )
            tdm_loss_student = tdm_loss / self.gradient_accumulation_steps
            tdm_loss_student.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                self.optimizer_student.step()
                self.optimizer_student.zero_grad()
                self.scheduler_student.step()

        # ===== Update Estimator (adversarial) =====
        if step % self.update_estimator_every == 0:
            self.student.eval()
            self.estimator.train()

            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    # Estimator tries to match Student's current distribution
                    with torch.no_grad():
                        student_noise_pred = self.student.predict_noise(
                            noisy_latents, timesteps, None, None
                        )

                    estimator_noise_pred = self.estimator.predict_noise(
                        noisy_latents, timesteps, None, None
                    )

                    # Estimator loss: match student's predictions
                    estimator_loss = nn.functional.mse_loss(estimator_noise_pred, student_noise_pred)
                    estimator_loss = estimator_loss / self.gradient_accumulation_steps

                self.scaler_estimator.scale(estimator_loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler_estimator.unscale_(self.optimizer_estimator)
                    torch.nn.utils.clip_grad_norm_(self.estimator.parameters(), max_norm=1.0)

                    self.scaler_estimator.step(self.optimizer_estimator)
                    self.scaler_estimator.update()
                    self.optimizer_estimator.zero_grad()
                    self.scheduler_estimator.step()
            else:
                # Without AMP
                with torch.no_grad():
                    student_noise_pred = self.student.predict_noise(
                        noisy_latents, timesteps, None, None
                    )

                estimator_noise_pred = self.estimator.predict_noise(
                    noisy_latents, timesteps, None, None
                )

                estimator_loss = nn.functional.mse_loss(estimator_noise_pred, student_noise_pred)
                estimator_loss = estimator_loss / self.gradient_accumulation_steps
                estimator_loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.estimator.parameters(), max_norm=1.0)
                    self.optimizer_estimator.step()
                    self.optimizer_estimator.zero_grad()
                    self.scheduler_estimator.step()
        else:
            estimator_loss = torch.tensor(0.0)

        return {
            "tdm_loss": tdm_loss.item(),
            "estimator_loss": estimator_loss.item() if isinstance(estimator_loss, torch.Tensor) else estimator_loss,
            "score_diff_norm": (epsilon_real - epsilon_fake).norm().item()
        }

    def train(self, dataloader):
        """
        Main training loop for TDM distillation.
        """
        print("Starting TDM training...")
        epochs = self.config.get("epochs", 100)

        for epoch in range(epochs):
            self.student.train()
            epoch_metrics = {
                "tdm_loss": 0.0,
                "estimator_loss": 0.0,
                "score_diff_norm": 0.0
            }

            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")

            for step, batch in pbar:
                metrics = self.train_step(batch, step)

                # Accumulate metrics
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]

                # Update progress bar
                pbar.set_postfix({
                    "TDM_loss": f"{metrics['tdm_loss']:.4f}",
                    "Est_loss": f"{metrics['estimator_loss']:.4f}",
                    "Score_diff": f"{metrics['score_diff_norm']:.4f}"
                })

            # Print epoch summary
            num_batches = len(dataloader)
            print(f"\nEpoch {epoch+1} Summary:")
            for key, value in epoch_metrics.items():
                print(f"  {key}: {value/num_batches:.4f}")

            # Save checkpoints
            if (epoch + 1) % self.config.get("save_every", 10) == 0:
                self.save_checkpoint(epoch + 1)

        print("Training complete!")

    def save_checkpoint(self, epoch):
        """Save model checkpoints (fp16 compatible)."""
        checkpoint = {
            "epoch": epoch,
            "student_state_dict": self.student.state_dict(),
            "estimator_state_dict": self.estimator.state_dict(),
            "optimizer_student_state_dict": self.optimizer_student.state_dict(),
            "optimizer_estimator_state_dict": self.optimizer_estimator.state_dict(),
            "config": self.config
        }

        # Also save scaler states if using AMP
        if self.use_amp:
            checkpoint["scaler_student_state_dict"] = self.scaler_student.state_dict()
            checkpoint["scaler_estimator_state_dict"] = self.scaler_estimator.state_dict()

        save_path = self.config.get("checkpoint_dir", "./checkpoints")
        torch.save(checkpoint, f"{save_path}/tdm_checkpoint_epoch_{epoch}.pt")
        print(f"Checkpoint saved: epoch {epoch}")

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint (fp16 compatible)."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.estimator.load_state_dict(checkpoint["estimator_state_dict"])
        self.optimizer_student.load_state_dict(checkpoint["optimizer_student_state_dict"])
        self.optimizer_estimator.load_state_dict(checkpoint["optimizer_estimator_state_dict"])

        # Load scaler states if they exist (for AMP)
        if self.use_amp and "scaler_student_state_dict" in checkpoint:
            self.scaler_student.load_state_dict(checkpoint["scaler_student_state_dict"])
            self.scaler_estimator.load_state_dict(checkpoint["scaler_estimator_state_dict"])

        # Ensure models are in fp16
        self.student.half()
        self.estimator.half()

        print(f"Checkpoint loaded from {checkpoint_path}")
