# Extract Method Generation using Deep Reinforcement Learning

## HPC Setup

### Basic Setup

- To check for available versions for any module and how to load them use this template `module spider <module_name>`
- Create `venv` in python. If you are facing difficulties in installing libraries with in-built dependency of let's say arrow, use `pip install --no-index <pkg_name>`

### Extra Setup (Specific for running trl library)

- First force purge all existing modules. Use this command `module --force purge`
- Load the `StdEnv/2023` since it comes bundled with latest `gcc (gcc/12.3)` and `python 3.11`. Load it using `module load StdEnv/2023`
- Load `arrow` using `module load arrow/14.0.1`

#### tl;dr for extra setup

Just execute the setup script like the following

```sh
source src/utils/setup_cc.sh
```

## Project Setup Steps

### Data Collection

If you want to collect data, follow the following steps:

- Download the `json` from [SEART](https://seart-ghs.si.usi.ch/) tool. 
- Place the `json` file in the input folder (data/input)
- Execute the `data-collector.sh` script like `source data-collector.sh` or if using HPC just use `sbatch data-collector.sh`.
- Alternatively, if you want to run the data collection script independently, execute the following:
```sh
python -u src/dataprocessing/data.py $input_file_name
```

### Data Pre Processing

- After running the data pre processing stage, it'll create a `jsonl` file with the before and after refactoring methods **for each repository**. 
- If running via HPC, the output will be a `zip` file present in the `data/output` foler. Extract it and the `jsonl` files will be in the `localstorage` folder.
- Just execute the `src/deep_learning/dataset_creation.py` script like following.

First, we need to collate all the data from different repository `jsonl` files to a single `jsonl` file.


```sh
python dataset_creation.py generate <input folder with all repository jsonl files> <output jsonl file path>
```

After generation, if you want to split the data, execute the following:

```sh
python dataset_creation.py split <jsonl file created in the last step> <output folder path>
```
With the collated input data and it'll create `train.jsonl`, `test.jsonl` and `val.jsonl`.


### Fine Tuning

- To execute the fine tuning script, you can run the `src/refactoring-finetune/ft-scripts/supervised_fine_tune.py` as follows:

```sh
python code-t5.py --model_save_path ./output/codet5-test --run_name code_t5_test --train_data_file_path data/dl-no-context-len/train.jsonl --eval_data_file_path data/dl-no-context-len/val.jsonl --num_epochs 
1
```

- This is just an example. Check out the `ScriptArguments` class in the file for more information on the arguments. Or run `python code-t5.py --help`.

Note: Make sure to setup `WandB` in the environment variable if you want to use W&B.

- If you want to run a batch job using HPC, just execute the following:

```sh
sbatch fine-tune.sh
```

### Reinforcement Leaning

- Move to the `src/reinforcement-learning` directory. 
- Run the `ppo_trl.py` script with the necessary arguments. An example is given below:

```sh
python ppo_trl.py 
--model_name src/refactoring-finetune/ft-scripts/output/code-t5-fine-tuned 
--tokenizer_name src/refactoring-finetune/ft-scripts/output/code-t5-fine-tuned 
--log_with wandb 
--train_data_file_path data/dl-large/preprocessed/train.jsonl 
--eval_data_file_path data/dl-large/preprocessed/val.jsonl 
```

- Check the `ScriptArguments` class in the file for more information.

- If you want to run a batch job using HPC, just execute the following:

```sh
sbatch rl-fine-tune.sh
```