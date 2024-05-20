#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=6:0:0
#SBATCH --gres=gpu:v100l:1
#SBATCH --signal=B:USR1@360
#SBATCH --mail-user=indranil.palit@dal.ca
#SBATCH --mail-type=ALL

project_location=`pwd`

source src/utils/setup_cc.sh

source .venv/bin/activate

cd src/refactoring-finetune/ft-scripts/

python code-t5.py --model_save_path ./output/codet5-ft --run_name code-t5-ft --train_data_file_path $project_location/data/dl-no-context-len/train.jsonl --eval_data_file_path $project_location/data/dl-no-context-len/val.jsonl --num_epochs 1