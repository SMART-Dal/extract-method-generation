#!/bin/bash

module --force purge

module load StdEnv/2023

module load arrow/14.0.1

module load java/17
module load maven/3.9.6

# export JAVA_TOOL_OPTIONS="-Xms256m -Xmx5g"
unset JAVA_TOOL_OPTIONS

# python -m venv .venv