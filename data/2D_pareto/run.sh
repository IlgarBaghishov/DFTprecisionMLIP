#!/bin/bash
#
#SBATCH -J 2Dpareto
#SBATCH -o output.txt
#SBATCH -e error_output.txt
#SBATCH -N 1
#SBATCH -t 48:00:00

python -u ../../get_2D_pareto_data.py
