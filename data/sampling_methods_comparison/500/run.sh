#!/bin/bash
#
#SBATCH -J 500sample
#SBATCH -o output.txt
#SBATCH -e error_output.txt
#SBATCH -N 1
#SBATCH -t 48:00:00

python -u ../../../get_sampling_performance_data.py 6 500
