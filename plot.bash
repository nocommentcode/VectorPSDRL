#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1
#$ -l h_rt=00:05:0
#$ -l h_vmem=4G
module load python/3.8.5
python plotting.py