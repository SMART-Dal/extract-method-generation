#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:2
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:0:0
#SBATCH --signal=B:USR1@180

python run_exp.py --model_tag codet5_small --task refactoring
