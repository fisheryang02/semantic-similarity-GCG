#!/bin/bash
export WANDB_MODE=disabled
# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'
export n=25
export model='llama2'
export device=0
export beta=1
export cosinedecay=True
export search=5
export select=5
export folder=llama2_with_cosdecay_and_lasthiddenstate
export similarity=llm
# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

export round=1

mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/launch_scripts
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/results

nohup python -u ../main.py \
    --config="../configs/transfer_llama2.py" \
    --config.attack=gcg \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.result_prefix="$HOME/llm-attacks/experiments/${folder}/round${round}/results/multiple_llama2_search_5_beta_1_select_5_cosinedecay_True_round_${round}" \
    --config.progressive_goals=False \
    --config.stop_on_success=False \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=$n \
    --config.n_steps=500 \
    --config.test_steps=50 \
    --config.batch_size=512 \
    --config.searchalpha=5 \
    --config.selectalpha=5 \
    --config.beta=1 \
    --config.cosine_decay=True \
    --config.similarity=llm \
    --config.devices[0]="cuda" > $HOME/llm-attacks/experiments/llama2_with_cosdecay_and_lasthiddenstate/round${round}/launch_scripts/multiple_llama2_search_5_beta_1_select_5_cosinedecay_True_round_${round}.log 2>&1 &
pid=$!
wait $pid

export round=2

mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/launch_scripts
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/results

nohup python -u ../main.py \
    --config="../configs/transfer_llama2.py" \
    --config.attack=gcg \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.result_prefix="$HOME/llm-attacks/experiments/${folder}/round${round}/results/multiple_llama2_search_5_beta_1_select_5_cosinedecay_True_round_${round}" \
    --config.progressive_goals=False \
    --config.stop_on_success=False \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=$n \
    --config.n_steps=500 \
    --config.test_steps=50 \
    --config.batch_size=512 \
    --config.searchalpha=5 \
    --config.selectalpha=5 \
    --config.beta=1 \
    --config.cosine_decay=True \
    --config.similarity=llm \
    --config.devices[0]="cuda" > $HOME/llm-attacks/experiments/llama2_with_cosdecay_and_lasthiddenstate/round${round}/launch_scripts/multiple_llama2_search_5_beta_1_select_5_cosinedecay_True_round_${round}.log 2>&1 &
pid=$!
wait $pid

export round=3

mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/launch_scripts
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/results

nohup python -u ../main.py \
    --config="../configs/transfer_llama2.py" \
    --config.attack=gcg \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.result_prefix="$HOME/llm-attacks/experiments/${folder}/round${round}/results/multiple_llama2_search_5_beta_1_select_5_cosinedecay_True_round_${round}" \
    --config.progressive_goals=False \
    --config.stop_on_success=False \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=$n \
    --config.n_steps=500 \
    --config.test_steps=50 \
    --config.batch_size=512 \
    --config.searchalpha=5 \
    --config.selectalpha=5 \
    --config.beta=1 \
    --config.cosine_decay=True \
    --config.similarity=llm \
    --config.devices[0]="cuda" > $HOME/llm-attacks/experiments/llama2_with_cosdecay_and_lasthiddenstate/round${round}/launch_scripts/multiple_llama2_search_5_beta_1_select_5_cosinedecay_True_round_${round}.log 2>&1 &
pid=$!
wait $pid
