#!/bin/bash
#SBATCH --job-name=slurmtest
#SBATCH --output=out_slurmtest
#SBATCH --error=myjob_error.log
#SBATCH --gres=gpu:0

cd /local/scratch/NETID/DIRECTORY   
source venv/bin/activate
python GSDOptimizer.py
