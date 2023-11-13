#!/bin/sh
#SBATCH --job-name=langsql
#SBATCH -t 72:00:00
#SBATCH -o slurm_logs/slurm-%j.out  # %j = job ID
#SBATCH --partition=gypsum-rtx8000
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem 30GB

python langchain_sqlcoder.py