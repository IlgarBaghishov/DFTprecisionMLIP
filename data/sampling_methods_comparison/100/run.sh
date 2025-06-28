#!/bin/bash
#
#SBATCH -J 100sample
#SBATCH -p amd-rome
#SBATCH --exclude=cn750,cn751,cn752,cn753,cn762,cn763
#SBATCH -o output.txt
#SBATCH -e error_output.txt
#SBATCH -N 1
#SBATCH --qos long
#SBATCH -t 48:00:00

python -u ../../../get_sampling_performance_data.py 100
