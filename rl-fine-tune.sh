#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=6:0:0
#SBATCH --gres=gpu:v100l:1
#SBATCH --signal=B:USR1@360
#SBATCH --mail-user=indranil.palit@dal.ca
#SBATCH --mail-type=ALL

project_location="/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation"

RUN_NAME={$1}
OUTPUT_FOLDER=${2}
MODEL_NAME=${3}
TOKENIZER_NAME=${4}

cd $project_location

source src/utils/setup_cc.sh

source .venv/bin/activate

cd src/reinforcement-learning/

python ppo_trl.py \
--model_name Salesforce/codet5-small \
--tokenizer_name Salesforce/codet5-small \
--log_with wandb \
--model_save_path ./ppo-output-pretrained \
--train_data_file_path /home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-large/preprocessed/len/train.jsonl \
--eval_data_file_path /home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/data/dl-large/preprocessed/len/val.jsonl 