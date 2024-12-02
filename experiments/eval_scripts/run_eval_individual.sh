#!/bin/bash

export LOG=$1
export d=$2
nohup python -u ../evaluate_individual.py \
    --config=../configs/transfer_vicuna.py \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.logfile="${LOG}" \
    --config.n_train_data=1 \
    --config.devices[0]="cuda:${d}"\
    --config.n_test_data=0 > eval_individual_100_optimizedcontrol.log 2>&1 &
