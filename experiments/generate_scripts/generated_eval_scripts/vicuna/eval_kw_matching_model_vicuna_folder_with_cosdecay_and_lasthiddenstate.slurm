#!/bin/bash
#SBATCH --job-name gcgeval_kw_matching_model_vicuna_folder__with_cosdecay_and_lasthiddenstate
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

srun bash batch_run_eval_kw_matching_model_vicuna_folder_with_cosdecay_and_lasthiddenstate_.sh

