#!/bin/bash

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $HOME/ieai-miniproject-context-mixing

source .venv/bin/activate
