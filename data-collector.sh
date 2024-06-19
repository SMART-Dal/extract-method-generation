#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=7-1:0:0
#SBATCH --signal=B:USR1@360
#SBATCH --mail-user=indranil.palit@dal.ca
#SBATCH --mail-type=ALL

echo "Start"

# Load modules
module purge
module load python/3.10
module load java/17
module load maven/3.6.3

# export JAVA_TOOL_OPTIONS="-Xms256m -Xmx5g"
unset JAVA_TOOL_OPTIONS

output_file_name=output-$(date +%Y-%m-%d-%H-%M-%S)
project_location=`pwd`
input_file_name=results.json

handle_signal() 
{
    echo 'Trapped signal'

    move_files
    
    exit 0
}

move_files() 
{
    echo 'Zipping output folder'
    zip -r $SLURM_TMPDIR/extract-method-generation/data/output/$output_file_name.zip $SLURM_TMPDIR/extract-method-generation/data/output
    echo 'Moving Files'
    rsync -axvH --no-g --no-p $SLURM_TMPDIR/extract-method-generation/data/output/$output_file_name.zip $project_location/data/output
}

trap 'handle_signal' SIGUSR1

cd $SLURM_TMPDIR
git clone --recursive git@github.com:IP1102/extract-method-generation.git

cd extract-method-generation/refminer-extractmethod
git checkout 172a9fe8f3fe59d41cc6b438094ac4210a0a3faa
mvn clean package
cd ..

# Move input file to the slurm folder
echo "Copying input file to the slurm folder"
rsync -axvH --no-g --no-p $project_location/data/input/$input_file_name $SLURM_TMPDIR/extract-method-generation/data/input/$input_file_name

python -u src/dataprocessing/data.py $input_file_name &

PID=$!
wait ${PID}


echo "Python Script execution over. Attempting to copy the output file..."

move_files

echo "Completed data collection process."



