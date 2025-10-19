#!/bin/bash
#SBATCH --job-name=analysis_lightcast
#SBATCH --partition=foodairesearch 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32GB
#SBATCH --output=/common/home/users/c/chhan/Work/lightcast/runs/%u.%j.out

python main.py