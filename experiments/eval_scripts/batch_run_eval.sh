#!/bin/bash
export model='vicuna'
export n=25
export eval_method='harmbench'
export device=1
export folder='vicuna_without_cosdecay_and_mean_sentemb'
demofun(){
    for file in `ls $1`
    do
        echo "file: $file"
        nohup python -u ../evaluate_${model}.py\
            --config="../configs/transfer_${model}.py"\
            --config.train_data="../../data/advbench/harmful_behaviors.csv"\
            --config.logfile="$1${file}"\
            --config.n_train_data=$n\
            --config.n_test_data=100\
    	    --config.eval_method=$eval_method\
            --config.devices[0]="cuda:${device}" > /home/yjj/project/llm-attacks/experiments/${folder}/round${round}/eval_scripts/${eval_method}/eval_${file}_eval_method_${eval_method}.log 2>&1 &
        pid=$!
        wait $pid
    done
}

export round=1
path="/home/yjj/project/llm-attacks/experiments/${folder}/round${round}/results/"
demofun $path

export round=2
path="/home/yjj/project/llm-attacks/experiments/${folder}/round${round}/results/"
demofun $path

# export round=3
# path="/home/yjj/project/llm-attacks/experiments/${folder}/round${round}/results/"
# demofun $path