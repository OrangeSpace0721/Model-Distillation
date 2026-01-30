"""
Main entry point for distillation experiments.
"""
import argparse
import torch
from src.models import Flux2Teacher, StudentModel
from src.distill import Distiller
from src.utils import get_dataloader
import yaml


def main():
    parser = argparse.ArgumentParser(description="Distill Flux.2 into smaller models.")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Assign teacher and student to different GPUs if available
    teacher_device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')
    student_device = torch.device('cuda:1' if torch.cuda.device_count() > 1 else teacher_device)
    config['teacher_device'] = teacher_device
    config['student_device'] = student_device

    teacher = Flux2Teacher(config).to(teacher_device)
    student = StudentModel(config).to(student_device)
    dataloader = get_dataloader(config)

    distiller = Distiller(teacher, student, dataloader, config)
    distiller.train()

if __name__ == "__main__":
    main()
