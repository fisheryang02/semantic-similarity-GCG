#!/bin/bash
export model='llama2'
export n=25
export eval_method='llamaguard'
export device=0
export folder='llama2_without_cosdecay_and_mean_sentemb'
demofun(){
    mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts/${eval_method}
    mkdir -p $HOME/llm-attacks/experiments/${folder}/round${round}/eval/${eval_method}
    for file in `ls $1`
    do
        echo "file: ${file}"
        echo `pwd`
        nohup python -u ../evaluate_llama2.py \
            --config="../configs/transfer_llama2.py"\
            --config.train_data="../../data/advbench/harmful_behaviors.csv" \
            --config.logfile="$1/${file}"\
            --config.n_train_data=$n\
            --config.n_test_data=100\
            --config.eval_method=${eval_method} \
            --config.devices[0]="cuda:${device}" > $HOME/llm-attacks/experiments/${folder}/round${round}/eval_scripts/${eval_method}/eval_${file}_eval_method_llamaguard.log 2>&1 &
        pid=$!
        wait $pid
    done
}

export round=1
path="$HOME/llm-attacks/experiments/${folder}/round${round}/results/"
demofun $path

export round=2
path="$HOME/llm-attacks/experiments/${folder}/round${round}/results/"
demofun $path

export round=3
path="$HOME/llm-attacks/experiments/${folder}/round${round}/results/"
demofun $path
