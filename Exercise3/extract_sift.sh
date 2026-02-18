#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=extract_sift
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=extract_sift%j.out
#SBATCH --error=extract_sift%j.err
#SBATCH --constraint=icx

source ~/venvs/cv_vlad/bin/activate
cd ~/Exercise3
export PYTHONPATH=$PWD:$PYTHONPATH

echo "=========================================="
echo "TASK E: Extracting SIFT descriptors"
echo "Date: $(date)"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

python skeleton.py \
                    --labels_train icdar17_labels_train.txt \
                    --labels_test icdar17_labels_test.txt \
                    --in_train icdar2017-training-binary \
                    --in_test ScriptNet-HistoricalWI-2017-binarized \
                    --suffix .png \
                    --powernorm \
                    --overwrite \
                    --extract-sift

echo ""
echo "SIFT extraction completed at $(date)"