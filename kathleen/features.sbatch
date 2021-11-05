#!/bin/bash 
#SBATCH --nodes=8                        # requests 8 compute servers
#SBATCH --ntasks-per-node=4              # runs 4 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=features
#SBATCH --output=features.out

source /scratch/ky2132/miniconda3/etc/profile.d/conda.sh
conda activate capstone_scratch
export PATH="/scratch/ky2132/miniconda3:$PATH"

python ./features.py
