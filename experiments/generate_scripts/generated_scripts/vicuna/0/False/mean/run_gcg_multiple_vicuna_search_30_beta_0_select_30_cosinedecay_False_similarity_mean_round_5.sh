#!/bin/bash
export WANDB_MODE=disabled
# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'
export n=25
export model='vicuna'
export device=0
export beta=1
export cosinedecay=False
export search=30
export select=30
export folder=vicuna_without_cosdecay_and_mean_sentemb
export similarity=mean
export round=5
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
    --config="../configs/transfer_vicuna.py" \
    --config.attack=gcg \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.result_prefix="$HOME/llm-attacks/experiments/${folder}/round${round}/results/multiple_vicuna_search_30_beta_0_select_30_cosinedecay_False_round_${round}" \
    --config.progressive_goals=False \
    --config.stop_on_success=False \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=$n \
    --config.n_steps=500 \
    --config.test_steps=50 \
    --config.batch_size=512 \
    --config.searchalpha=30 \
    --config.selectalpha=30 \
    --config.beta=1 \
    --config.cosine_decay=False \
    --config.similarity=mean \
    --config.devices[0]="cuda" > $HOME/llm-attacks/experiments/vicuna_without_cosdecay_and_mean_sentemb/round${round}/launch_scripts/multiple_vicuna_search_30_beta_0_select_30_cosinedecay_False_round_${round}.log 2>&1 &
pid=$!
wait $pid



export eval_method='harmbench'
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts/${eval_method}
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval/${eval_method}

nohup python -u ../evaluate_vicuna.py \
    --config="../configs/transfer_vicuna.py"\
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.logfile="$HOME/llm-attacks/experiments/${folder}/round${round}/results/multiple_vicuna_search_30_beta_0_select_30_cosinedecay_False_round_${round}.json"\
    --config.n_train_data=$n\
    --config.n_test_data=100\
    --config.eval_method=${eval_method} \
    --config.devices[0]="cuda:${device}" > $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts/${eval_method}/eval_${file}_eval_method_${eval_method}.log 2>&1 &
pid=$!
wait $pid

export eval_method='kw_matching'
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts/${eval_method}
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval/${eval_method}

nohup python -u ../evaluate_vicuna.py \
    --config="../configs/transfer_vicuna.py"\
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.logfile="$HOME/llm-attacks/experiments/${folder}/round${round}/results/multiple_vicuna_search_30_beta_0_select_30_cosinedecay_False_round_${round}.json"\
    --config.n_train_data=$n\
    --config.n_test_data=100\
    --config.eval_method=${eval_method} \
    --config.devices[0]="cuda:${device}" > $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts/${eval_method}/eval_${file}_eval_method_${eval_method}.log 2>&1 &
pid=$!
wait $pid

eval "$(conda shell.bash hook)"
conda activate gcgeval
export eval_method='llamaguard'
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts/${eval_method}
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval/${eval_method}

nohup python -u ../evaluate_vicuna.py \
    --config="../configs/transfer_vicuna.py"\
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.logfile="$HOME/llm-attacks/experiments/${folder}/round${round}/results/multiple_vicuna_search_30_beta_0_select_30_cosinedecay_False_round_${round}.json"\
    --config.n_train_data=$n\
    --config.n_test_data=100\
    --config.eval_method=${eval_method} \
    --config.devices[0]="cuda:${device}" > $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts/${eval_method}/eval_${file}_eval_method_${eval_method}.log 2>&1 &
pid=$!
wait $pid
