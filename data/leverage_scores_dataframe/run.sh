#!/bin/bash
#
#SBATCH -J lev_scores
#SBATCH -o output.txt
#SBATCH -e error_output.txt
#SBATCH -N 1
#SBATCH -t 48:00:00

python -u ../../get_leverage_scores_data.py 6
