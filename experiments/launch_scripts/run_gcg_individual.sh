#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=$1 # llama2 or vicuna
export setup=$2 # behaviors or strings
export n_train_data=$3
export device=$4
# Create results folder if it doesn't exist
if [ ! -d "../results" ]
then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

for offset in 99
do
    python -u ../main.py \
        --config="../configs/individual_${model}.py" \
        --config.attack=gcg \
        --config.train_data="../../data/advbench/harmful_${setup}.csv" \
        --config.result_prefix="../results/individual_${setup}_${model}_${offset}_gcg" \
        --config.data_offset=${offset}\
        --config.n_train_data=${n_train_data} \
        --config.n_steps=1000 \
        --config.test_steps=50 \
        --config.devices[0]="cuda:${device}"\
        --config.alpha=0\
        --config.batch_size=512 > individual_${model}_${setup}_${offset}.log 2>&1 &
done
        

#Experiments
#The experiments folder contains code to reproduce GCG experiments on AdvBench.
#
#To run individual experiments with harmful behaviors and harmful strings (i.e. 1 behavior, 1 model or 1 string, 1 model), run the following code inside experiments (changing vicuna to llama2 and changing behaviors to strings will switch to different experiment setups):
#cd launch_scripts
#bash run_gcg_individual.sh vicuna behaviors
#To perform multiple behaviors experiments (i.e. 25 behaviors, 1 model), run the following code inside experiments:
#cd launch_scripts
#bash run_gcg_multiple.sh vicuna # or llama2
#To perform transfer experiments (i.e. 25 behaviors, 2 models), run the following code inside experiments:
#cd launch_scripts
#bash run_gcg_transfer.sh vicuna 2 # or vicuna_guanaco 4
#To perform evaluation experiments, please follow the directions in experiments/parse_results.ipynb.