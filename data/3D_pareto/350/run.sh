#!/bin/bash
#
#SBATCH -J 350_3Dpareto
#SBATCH -p amd-rome
#SBATCH --exclude=cn750,cn751,cn752,cn753,cn762,cn763
#SBATCH -o output.txt
#SBATCH -e error_output.txt
#SBATCH -N 1
#SBATCH --qos long
#SBATCH -t 48:00:00

python -u ../../../get_3D_pareto_data.py 50 350
