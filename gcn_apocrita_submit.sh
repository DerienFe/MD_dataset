#!/bin/bash
#$ -cwd
#$ -pe smp 8
#$ -l h_vmem=7.5G
#$ -l h_rt=12:0:0
#$ -l gpu=1

export CUDA_VISIBLE_DEVICES=${SGE_HGR_gpu// /,}
module load python
virtualenv --include-lib pytorchenv
source pytorchenv/bin/activate
pip install torch torchvision scipy numpy sklearn


source ~/pytorchenv/bin/activate
python GCN.py > pythonrun.log