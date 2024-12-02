#!/bin/bash
export n=25
export model=vicuna
export device=0
nohup python -u ../evaluate_${model}.py\
    --config="../configs/transfer_${model}.py"\
    --config.train_data="../../data/advbench/harmful_behaviors.csv"\
    --config.logfile="/home/yjj/project/llm-attacks/experiments/results/multiple_vicuna_search_35_beta_1_select_35_cosinedecay_True_round_2_20240723-02:25:21.json"\
    --config.n_train_data=$n\
    --config.n_test_data=100\
    --config.eval_method="harmbench"\
    --config.devices[0]="cuda:${device}" > "multiple_vicuna_search_35_beta_1_select_35_cosinedecay_True_round_2_20240723-02:25:21.json10086.log" 2>&1
    