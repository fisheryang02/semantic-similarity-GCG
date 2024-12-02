#!/bin/bash
export WANDB_MODE=disabled
# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'
export n=25
export model='llama2'
export device=0
export beta=1
export cosinedecay=True
export search=15
export select=15
export folder=llama2_with_cosdecay_and_mean_sentemb
export similarity=mean
export round=2
# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/launch_scripts
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/results

nohup python -u ../main.py \
    --config="../configs/transfer_llama2.py" \
    --config.attack=gcg \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.result_prefix="$HOME/llm-attacks/experiments/${folder}/round${round}/results/multiple_llama2_search_15_beta_1_select_15_cosinedecay_True_round_${round}" \
    --config.progressive_goals=False \
    --config.stop_on_success=False \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=$n \
    --config.n_steps=500 \
    --config.test_steps=50 \
    --config.batch_size=512 \
    --config.searchalpha=15 \
    --config.selectalpha=15 \
    --config.beta=1 \
    --config.cosine_decay=True \
    --config.similarity=mean \
    --config.devices[0]="cuda" > $HOME/llm-attacks/experiments/llama2_with_cosdecay_and_mean_sentemb/round${round}/launch_scripts/multiple_llama2_search_15_beta_1_select_15_cosinedecay_True_round_${round}.log 2>&1 &
pid=$!
wait $pid
