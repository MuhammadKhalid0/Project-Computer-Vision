#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=task_f1_gmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=task_f1_gmp_%j.out
#SBATCH --error=task_f1_gmp_%j.err
#SBATCH --constraint=icx

source ~/venvs/cv_vlad/bin/activate
cd ~/Exercise3
export PYTHONPATH=$PWD:$PYTHONPATH

echo "=========================================="
echo "TASK F: GMP + power norm"
echo "Date: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

for gamma in 0.001 0.01 0.1 1 10 50 100; do
    python skeleton.py \
                     --labels_train icdar17_labels_train.txt \
                    --labels_test icdar17_labels_test.txt \
                    --in_train icdar2017-sift-train \
                    --in_test icdar2017-sift-test \
                    --suffix _SIFT_patch_pr.pkl.gz \
                    --powernorm \
                    --gmp \
                    --gamma $gamma \
                    --overwrite
        2>&1 | tee results_gmp_gamma${gamma}.txt
done

echo ""
echo "Task F completed at $(date)"