#!/bin/bash
export WANDB_MODE=disabled
# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'
export n=25
export model='llama2'
export device=0
export beta=1
export cosinedecay=True
export search=40
export select=40
export folder=llama2_with_cosdecay_and_lasthiddenstate
export similarity=llm
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
    --config="../configs/transfer_llama2.py" \
    --config.attack=gcg \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.result_prefix="$HOME/llm-attacks/experiments/${folder}/round${round}/results/multiple_llama2_search_40_beta_1_select_40_cosinedecay_True_round_${round}" \
    --config.progressive_goals=False \
    --config.stop_on_success=False \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=$n \
    --config.n_steps=500 \
    --config.test_steps=50 \
    --config.batch_size=512 \
    --config.searchalpha=40 \
    --config.selectalpha=40 \
    --config.beta=1 \
    --config.cosine_decay=True \
    --config.similarity=llm \
    --config.devices[0]="cuda" > $HOME/llm-attacks/experiments/llama2_with_cosdecay_and_lasthiddenstate/round${round}/launch_scripts/multiple_llama2_search_40_beta_1_select_40_cosinedecay_True_round_${round}.log 2>&1 &
pid=$!
wait $pid



export eval_method='harmbench'
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts/${eval_method}
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval/${eval_method}

nohup python -u ../evaluate_llama2.py \
    --config="../configs/transfer_llama2.py"\
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.logfile="$HOME/llm-attacks/experiments/${folder}/round${round}/results/multiple_llama2_search_40_beta_1_select_40_cosinedecay_True_round_${round}.json"\
    --config.n_train_data=$n\
    --config.n_test_data=100\
    --config.eval_method=${eval_method} \
    --config.devices[0]="cuda:${device}" > $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts/${eval_method}/eval_${file}_eval_method_${eval_method}.log 2>&1 &
pid=$!
wait $pid

export eval_method='kw_matching'
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts/${eval_method}
mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval/${eval_method}

nohup python -u ../evaluate_llama2.py \
    --config="../configs/transfer_llama2.py"\
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.logfile="$HOME/llm-attacks/experiments/${folder}/round${round}/results/multiple_llama2_search_40_beta_1_select_40_cosinedecay_True_round_${round}.json"\
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

nohup python -u ../evaluate_llama2.py \
    --config="../configs/transfer_llama2.py"\
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.logfile="$HOME/llm-attacks/experiments/${folder}/round${round}/results/multiple_llama2_search_40_beta_1_select_40_cosinedecay_True_round_${round}.json"\
    --config.n_train_data=$n\
    --config.n_test_data=100\
    --config.eval_method=${eval_method} \
    --config.devices[0]="cuda:${device}" > $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts/${eval_method}/eval_${file}_eval_method_${eval_method}.log 2>&1 &
pid=$!
wait $pid

