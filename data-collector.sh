#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=3:0:0
#SBATCH --signal=B:USR1@360

echo "Start"

module load python/3.10
module load java/17
module load maven/3.6.3

export JAVA_TOOL_OPTIONS="-Xms256m -Xmx5g"

output_file_name=output-$(date +%Y-%m-%d-%H-%M-%S)
project_location=/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation

handle_signal() 
{
    echo 'Trapped signal'
    echo 'Zipping output folder'
    zip -r $SLURM_TMPDIR/extract-method-generation/data/output/$output_file_name.zip $project_location/data/output
    
    echo 'Moving File'
    rsync -axvH --no-g --no-p $SLURM_TMPDIR/extract-method-generation/data/output/$output_file_name.zip $project_location/data/output
    exit 0
}

trap 'handle_signal' SIGUSR1

cd $SLURM_TMPDIR
git clone --recursive git@github.com:IP1102/extract-method-generation.git

cd extract-method-generation/refminer-extractmethod

mvn clean package
cd ..


python -u src/dataprocessing/data.py &

PID=$!
wait ${PID}


echo "Python Script execution over. Attempting to copy the output file..."

echo 'Zipping output folder'
zip -r $SLURM_TMPDIR/extract-method-generation/data/output/$output_file_name.zip $project_location/data/output

echo 'Moving File'
rsync -axvH --no-g --no-p $SLURM_TMPDIR/extract-method-generation/data/output/$output_file_name.zip $project_location/data/output


echo "Completed data collection process."


