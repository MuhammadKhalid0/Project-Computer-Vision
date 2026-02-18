#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=task_osr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=task_osr_%j.out
#SBATCH --error=task_osr_%j.err
#SBATCH --constraint=icx

source ~/exercise4/osr/bin/activate
cd ~/exercise4

echo "=========================================="
echo "TASK OSR"
echo "Date: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

PYTHONPATH=src python3 hyperparameter_tuning.py

echo ""
echo "Task OSR completed at $(date)" 