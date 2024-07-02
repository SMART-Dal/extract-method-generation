#!/bin/bash

module --force purge

module load StdEnv/2023

module load arrow/14.0.1

module load java/17
module load maven/3.9.6

# export JAVA_TOOL_OPTIONS="-Xms256m -Xmx5g"
unset JAVA_TOOL_OPTIONS

# python -m venv .venv

# Make sure to run this in the job node (salloc terminal in interactive)
if [ -n "$SLURM_TMPDIR" ]; then
    if [ -d "$SLURM_TMPDIR/rl-template" ]; then
        echo "rl-template already exists in $SLURM_TMPDIR"
    else
        # If rl-template doesn't exist, copy it to $SLURM_TMPDIR
        cp -r /home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/rl-template/ "$SLURM_TMPDIR"
        echo "rl-template copied to $SLURM_TMPDIR"
    fi
else
    echo "SLURM_TMPDIR is empty"
fi