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

cd src/reinforcement-learning/

source ../utils/setup_cc.sh

source ../../.venv/bin/activate

cd playground/

python llama.py $project_location/data/dl-no-context/train.jsonl $project_location/data/dl-no-context/val.jsonl

