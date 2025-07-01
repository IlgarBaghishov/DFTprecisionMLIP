#!/bin/bash
#
#SBATCH -J 5000_3Dpareto
#SBATCH -o output.txt
#SBATCH -e error_output.txt
#SBATCH -N 1
#SBATCH -t 48:00:00

python -u ../../../get_3D_pareto_data.py 35 5000
