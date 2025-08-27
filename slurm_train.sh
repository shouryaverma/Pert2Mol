#!/usr/bin/env bash

#!/bin/bash
#SBATCH -A pccr
#SBATCH -N 1
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -t 48:00:00
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=14
#SBATCH --mail-user=verma198@purdue.edu
#SBATCH --mail-type=FAIL

starts=$(date +"%s")
start=$(date +"%r, %m-%d-%Y")

module load conda
conda activate /depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/src_new/ldmol_newenv
# source $HOME/.bashrc
export CUDA_LAUNCH_BLOCKING=1

# Set GPU environment variable for multi-GPU training
export GPU=0,1,2,3

# Change to your project directory
cd /depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/src_new

# Run the multi-GPU training pipeline
CUDA_VISIBLE_DEVICES=0,1,2,3 python src_new/LDMol/train/train_ldmol.py

ends=$(date +"%s")
end=$(date +"%r, %m-%d-%Y")
diff=$(($ends-$starts))
hours=$(($diff / 3600))
dif=$(($diff % 3600))
minutes=$(($dif / 60))
seconds=$(($dif % 60))

printf "\n\t===========Time Stamp===========\n"
printf "\tStart\t:$start\n\tEnd\t:$end\n\tTime\t:%02d:%02d:%02d\n" "$hours" "$minutes" "$seconds"
printf "\t================================\n\n"

sacct --jobs=$SLURM_JOBID --format=jobid,jobname,qos,nnodes,ncpu,maxrss,cputime,avecpu,elapsed