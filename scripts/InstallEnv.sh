#!/bin/bash

module purge
module load 2022
module load Anaconda3/2022.05

cd $HOME/ieai-miniproject-context-mixing/
conda env remove -n ieai-context-mix
conda env create -f environment.yaml
source activate ieai-context-mix
