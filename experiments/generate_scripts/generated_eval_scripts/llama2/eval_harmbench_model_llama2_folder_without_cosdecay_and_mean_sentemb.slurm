#!/bin/bash
#SBATCH --job-name gcgeval_harmbench_model_llama2_folder__without_cosdecay_and_mean_sentemb
#SBATCH --partition critical
#SBATCH -n 2
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16GB
#SBATCH --gres gpu:NVIDIAA40:1
#SBATCH --time 5-00:00:00
#SBATCH --mail-user 794205443@qq.com
#SBATCH --mail-type ALL
#SBATCH --output log.log
#SBATCH -A wangwj1-critical

srun bash batch_run_eval_harmbench_model_llama2_folder_without_cosdecay_and_mean_sentemb_.sh

