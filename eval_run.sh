#!/bin/bash

# This is a script for submitting pfn training jobs on the HPC3 bias.

#SBATCH --job-name=eval_efn                  ## Name of the job.
#SBATCH -A daniel_lab_gpu                                 ## account to charge 
#SBATCH -p gpu                                   ## partition/queue name
# #SBATCH --gres=gpu:V100:1
#SBATCH --nodes=1                                 ## (-N) number of nodes to use
#SBATCH --mem=60G                                 ## Request 3GB of memory per task
#SBATCH --time=0-06:00:00

#SBATCH --error=./outfiles/%x_%a.err       ## error log file
#SBATCH --output=./outfiles/%x_%a.out      ## output file

# Start by printing out hostname and date of job
echo "Found a node, here's some info: "
hostname; date
echo "================================"

# Setup environment
echo "Building software environment"
source ~/.bashrc
module load anaconda/2022.05
conda activate toptag
module load tensorflow/2.8.0

# Run command
./evaluate_all.sh ./checkpoints/29-0.45.tf efn
echo -e "\n\nDone!"
