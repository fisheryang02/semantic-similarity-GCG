#!/bin/bash
#SBATCH --job-name gcg0llama2          
#SBATCH --partition critical
#SBATCH -n 2
#SBATCH --nodes 1
#SBATCH --ntasks 1                  
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16GB
#SBATCH --gres gpu:NVIDIAA40:1
#SBATCH --time 5-00:00:00
#SBATCH --output log.log
#SBATCH --mail-user 17755923416@163.com
#SBATCH --mail-type ALL
#SBATCH -A wangwj1-critical

srun bash run_gcg_multiple_llama2_search_0_beta_0_select_0_cosinedecay_True_similarity_llm_round_4.sh
