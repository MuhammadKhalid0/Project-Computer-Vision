#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=task_g1_multivlad_WithoutPowerNorm_NoGMP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=task_g1_multivlad_WithoutPowerNorm_NoGMP_%j.out
#SBATCH --error=task_g1_multivlad_WithoutPowerNorm_NoGMP_%j.err
#SBATCH --constraint=icx

source ~/venvs/cv_vlad/bin/activate
cd ~/Exercise3
export PYTHONPATH=$PWD:$PYTHONPATH

echo "=========================================="
echo "TASK G1: Multi-VLAD (5 codebooks) + PCA whitening (without E-SVM)"
echo "Date: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

python skeleton.py \
  --labels_train icdar17_labels_train.txt \
  --labels_test icdar17_labels_test.txt \
  --in_train icdar2017-sift-train \
  --in_test icdar2017-sift-test \
  --suffix _SIFT_patch_pr.pkl.gz \
  --multivlad \
  --pca_components 1000 \
  --overwrite

echo ""
echo "Task G completed at $(date)"