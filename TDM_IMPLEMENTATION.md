# TDM (Trajectory Distribution Matching) Implementation

## Overview

This codebase implements **Trajectory Distribution Matching (TDM)**, the state-of-the-art method for distilling large diffusion models (like Flux, SDXL) into fast 1-4 step generators.

**All models use fp16 (float16) precision for maximum memory efficiency and speed.**

## Architecture

### The Three-Model System

TDM uses a dual-model training loop with three components:

| Model | Status | Role |
|-------|--------|------|
| **Teacher** | Frozen | Original pretrained diffusion model. Provides the "Real Score" (ground truth gradient towards valid data) |
| **Student** | Trainable | Copy of Teacher, learns to traverse the diffusion trajectory in just 1-4 steps |
| **Estimator** | Trainable | Copy of Teacher, acts as adversary/critic, learns to predict the Student's CURRENT output distribution |

### Key Files

- **[src/models.py](src/models.py)**: Model definitions
  - `Flux2Teacher`: Frozen teacher model wrapper
  - `StudentModel`: Trainable student that learns fast inference
  - `ScoreEstimator`: Trainable adversary for distribution matching

- **[src/distill.py](src/distill.py)**: TDM training loop
  - `TDMDistiller`: Main distillation class with TDM algorithm implementation

- **[src/utils.py](src/utils.py)**: Utilities
  - `NoiseScheduler`: Handles diffusion forward process
  - `TDMWeightScheduler`: Loss weighting for different timesteps

- **[train_tdm.py](train_tdm.py)**: Example training script

## The TDM Algorithm

### Training Loop (Each Step)

```
1. Sample random noise z ~ N(0, I)
2. Generate fake sample with Student: x̂ = Student(z)
3. Add noise to get noisy latents: x_t = √(ᾱ_t)·x̂ + √(1-ᾱ_t)·ε
4. Compute Real Score:    ε_real  = Teacher(x_t, t)    [frozen, ground truth]
5. Compute Fake Score:    ε_fake  = Estimator(x_t, t)  [trainable, adversary]
6. Compute TDM Loss:      L_TDM   = w(t) · ||ε_real - ε_fake||²
7. Update Student:        ∇_θ L_TDM  (push towards Real, away from Fake)
8. Update Estimator:      Match Student's current predictions
```

### The Magic of TDM

The key insight is in **Step 6**: The TDM loss is the **difference** between Real and Fake scores:

- If `ε_real ≈ ε_fake`: Student's distribution matches Teacher's → Loss ≈ 0
- If they differ: Gradient forcefully corrects the Student's trajectory

This solves the "blurriness vs. consistency" trade-off:
- **Trajectory Matching**: Student follows Teacher's valid ODE path
- **Distribution Matching**: Estimator prevents "regression to the mean" (blurriness)

## Usage

### Basic Training

```python
from src.distill import TDMDistiller
from src.utils import get_dataloader

config = {
    "flux_teacher_path": "./flux2_teacher",
    "epochs": 100,
    "batch_size": 4,  # Can be larger with fp16!
    "lr_student": 1e-5,
    "lr_estimator": 1e-5,
    "student_steps": 4,  # 1-4 step generation
    "use_amp": True,  # Automatic mixed precision with fp16
    "device": "cuda"
}

dataloader = get_dataloader(config)
distiller = TDMDistiller(config)
distiller.train(dataloader)
```

### Using the Training Script

```bash
python train_tdm.py
```

## Configuration Parameters

### Model Configuration

- `flux_teacher_path`: Path to pretrained Flux model
- `student_steps`: Number of inference steps (1-4)

### Training Hyperparameters

- `lr_student`: Learning rate for Student (default: 1e-5)
- `lr_estimator`: Learning rate for Estimator (default: 1e-5)
- `batch_size`: Batch size (can be larger with fp16! ~2x more than fp32)
- `epochs`: Number of training epochs
- `gradient_accumulation_steps`: Accumulate gradients over N steps
- `use_amp`: Use automatic mixed precision (default: True, highly recommended)

### TDM-Specific Parameters

- `weight_type`: Loss weighting strategy
  - `"constant"`: Uniform weight across all timesteps
  - `"snr"`: Signal-to-noise ratio weighting
  - `"min_snr"`: Min-SNR strategy (from "Min-SNR Weighting" paper)

- `update_estimator_every`: Update Estimator every N steps (default: 1)
- `use_student_generation`: Use Student-generated samples for training (recommended: True)

### Noise Scheduling

- `num_train_timesteps`: Number of diffusion timesteps (default: 1000)
- `beta_start`: Starting noise level (default: 0.0001)
- `beta_end`: Ending noise level (default: 0.02)
- `schedule_type`: Noise schedule type ("linear" or "scaled_linear")

## Key Implementation Details

### 1. Dual Optimizer Setup

```python
optimizer_student = AdamW(student.parameters(), lr=1e-5)
optimizer_estimator = AdamW(estimator.parameters(), lr=1e-5)
```

Both models train simultaneously but with different objectives.

### 2. Score Computation

```python
# Real Score (Teacher - frozen)
with torch.no_grad():
    epsilon_real = teacher.predict_noise(noisy_latents, timesteps)

# Fake Score (Estimator - trainable)
epsilon_fake = estimator.predict_noise(noisy_latents, timesteps)
```

### 3. TDM Loss

```python
# Weighted difference between Real and Fake scores
weights = weight_scheduler.get_weight(timesteps)
score_difference = epsilon_real - epsilon_fake
loss = weights * (score_difference ** 2)
```

### 4. Estimator Update (Adversarial)

```python
# Estimator tries to match Student's current predictions
with torch.no_grad():
    student_pred = student.predict_noise(noisy_latents, timesteps)

estimator_pred = estimator.predict_noise(noisy_latents, timesteps)
estimator_loss = mse_loss(estimator_pred, student_pred)
```

## Advantages of TDM

1. **State-of-the-Art Quality**: Best quality for 1-4 step generation
2. **No Mode Collapse**: Adversarial training prevents blurry outputs
3. **Trajectory Consistency**: Follows Teacher's valid ODE path
4. **Flexible**: Works with any diffusion model (SDXL, Flux, Video models)
5. **Memory Efficient**: fp16 precision reduces memory usage by ~50%
6. **Fast Training**: fp16 + AMP provides ~2-3x speedup over fp32

## Training Tips

1. **GPU Memory**: TDM requires 3x model memory (Teacher + Student + Estimator)
   - **fp16 reduces memory by ~50%** compared to fp32!
   - With fp16, you can use ~2x larger batch sizes
   - Use gradient checkpointing if still needed
   - Use gradient accumulation for larger effective batch sizes

2. **Mixed Precision (AMP)**:
   - **Always use `use_amp=True`** (automatic mixed precision with GradScaler)
   - Provides better training stability with fp16
   - Automatically handles gradient scaling to prevent underflow
   - Minimal overhead, maximum benefit

3. **Learning Rates**: Start with equal learning rates (1e-5)
   - If Student diverges: Decrease `lr_student`
   - If Estimator doesn't keep up: Increase `lr_estimator`
   - fp16 training is generally more stable than fp32

4. **Estimator Update Frequency**:
   - Update every step (`update_estimator_every=1`) for best quality
   - Update less frequently to save compute

5. **Weight Scheduling**:
   - Start with `"snr"` weighting
   - Try `"min_snr"` if training is unstable

6. **Student Steps**:
   - Start with 4 steps for easier training
   - Gradually reduce to 2 or 1 step if needed

## Monitoring Training

Key metrics to watch:

- **TDM Loss**: Should decrease over time
  - If stuck: Check learning rates
  - If exploding: Reduce learning rates or add gradient clipping

- **Score Difference**: `||ε_real - ε_fake||`
  - Should decrease as Student matches Teacher
  - If not decreasing: Estimator may not be keeping up

- **Estimator Loss**: Estimator's ability to track Student
  - Should stay low and stable
  - If increasing: Student is changing too fast

## References

This implementation is based on the TDM methodology described in:
- "Trajectory Distribution Matching for Diffusion Distillation" (2026)
- Key insight: Unifying trajectory matching with adversarial distribution matching

## Next Steps

1. Replace `FakeData` in `utils.py` with real dataset (e.g., LAION, ImageNet)
2. Add text conditioning support (encoder_hidden_states)
3. Implement better solvers (DPM-Solver, DDIM) for Student generation
4. Add validation/sampling during training to monitor quality
5. Implement mixed precision training (FP16/BF16) for faster training
