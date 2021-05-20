#!/bin/bash

#SBATCH --job-name=params          # Job name
#SBATCH --output=params.%j         # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=6          # Schedule one core
#SBATCH --mem=64G                  # Memory
#SBATCH --gres=gpu:1               # Schedule a GPU
#SBATCH --time=3-00:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=red		       # Run on either the Red or Brown queue

module load Python/3.7.4-GCCcore-8.3.0
pip3 install --user datasets transformers carbontracker deepspeed
pip3 install --user torch torchvision pymongo


python3 opt_csv.py $SLURM_JOB_ID $1
