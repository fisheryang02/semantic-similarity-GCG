#!/bin/bash
export WANDB_MODE=disabled
# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'
export n=25
export model='vicuna' # llama2 or vicuna
export device=0
export beta=0
export cosine_decay=True
export similarity='mean'
# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

for alpha in 20
do
    python -u ../main.py \
        --config="../configs/transfer_${model}.py" \
        --config.attack=gcg \
        --config.train_data="../../data/advbench/harmful_behaviors.csv" \
        --config.result_prefix="../results/multiple_${model}_gcg_${n}_alpha_${alpha}" \
        --config.progressive_goals=False \
        --config.stop_on_success=False \
        --config.num_train_models=1 \
        --config.allow_non_ascii=False \
        --config.n_train_data=$n \
        --config.n_steps=500 \
        --config.test_steps=50 \
        --config.batch_size=512\
        --config.searchalpha=${alpha}\
        --config.selectalpha=${alpha}\
        --config.beta=${beta}\
        --config.cosine_decay=${cosine_decay}\
        --config.similarity=${similarity}\
        --config.devices[0]="cuda:${device}" #> multiple_${model}_alpha_${alpha}_beta_${beta}.log 2>&1 &
done
