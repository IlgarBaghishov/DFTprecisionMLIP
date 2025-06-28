#!/bin/bash
#
#SBATCH -J get_A_b
#SBATCH -p amd-rome
#SBATCH --exclude=cn750,cn751,cn752,cn753,cn762,cn763
#SBATCH -o output.txt
#SBATCH -e error_output.txt
#SBATCH -N 1
#SBATCH --qos long
#SBATCH -t 4:00:00

mpirun -np 48 python ../../get_matrices_for_fitting.py 4 5
