#!/bin/bash
#SBATCH --job-name=distill_flux2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load python/3.10 cuda/11.7
source ~/envs/flux2/bin/activate

cd $SLURM_SUBMIT_DIR

python main.py --config configs/example.yaml
