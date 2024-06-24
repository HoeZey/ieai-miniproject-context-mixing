#!/bin/bash

module purge
module load 2022
module load Anaconda3/2022.05

cd $HOME/ieai-miniproject-context-mixing/
source activate ieai-context-mix
conda list
