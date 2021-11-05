#!/bin/bash 
#SBATCH --nodes=4                        # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=create_segments
#SBATCH --output=create_segments.out

#module load python/intel/3.8.6
source /scratch/ky2132/miniconda3/etc/profile.d/conda.sh
conda activate capstone_scratch
export PATH="/scratch/ky2132/miniconda3:$PATH"

python ./create_segments.py
