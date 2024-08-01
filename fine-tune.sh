#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=8:0:0
#SBATCH --gres=gpu:v100l:1
#SBATCH --signal=B:USR1@360
#SBATCH --mail-user=indranil.palit@dal.ca
#SBATCH --mail-type=ALL

project_location="/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation"

RUN_NAME={$1}
OUTPUT_FOLDER=${2}
MODEL_NAME=${3}
TOKENIZER_NAME=${4}
TRAIN_DATA_FILE_RELATIVE_PATH=${5}
EVAL_DATA_FILE_RELATIVE_PATH=${6}
NUMBER_OF_EPOCHS=${7}

cd $project_location

source src/utils/setup_cc.sh

source .venv/bin/activate

cd src/refactoring-finetune/ft-scripts/

python supervised_fine_tune.py \
--model_name $MODEL_NAME \
--tokenizer_name $TOKENIZER_NAME \
--model_save_path $OUTPUT_FOLDER \
--run_name $RUN_NAME \
--train_data_file_path $project_location/$TRAIN_DATA_FILE_RELATIVE_PATH \
--eval_data_file_path $project_location/$EVAL_DATA_FILE_RELATIVE_PATH \
--num_epochs $NUMBER_OF_EPOCHS

# sbatch fine-tune.sh code-t5-19k-15 ./output/code-t5-19k-15 Salesforce/codet5-small Salesforce/codet5-small data/dl-large/preprocessed/train.jsonl data/dl-large/preprocessed/val.jsonl 15