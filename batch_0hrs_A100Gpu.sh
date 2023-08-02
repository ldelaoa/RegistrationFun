#!/bin/bash
#SBATCH --time=0-00:48:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100.20gb:1
#SBATCH --mem=80GB
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
source /home1/p308104/Segmentation_Ch2/Subfolder/Envs/SegmentationCh2_Env1/bin/activate
python -u ./init.py
