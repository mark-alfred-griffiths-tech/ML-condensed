#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10GB
#SBATCH --job-name=log_reg
#SBATCH --partition=partition-1
#SBATCH --nodes=1
#SBATCH --output=/users/k1754828/log/core_analysis/preprocessing/preprocessing_phase_one/add_new_regressors/assemble/assemble_cf/assemble_cf-%j.output
#SBATCH --error=/users/k1754828/log/core_analysis/preprocessing/preprocessing_phase_one/add_new_regressors/assemble/assemble_cf/assemble_cf-%j.error

wd=/users/k1754828/ml_stuttering_project/core_analysis/preprocessing/preprocessing_phase_one/add_new_regressors/assemble/assemble_cf/

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710
cd $wd || exit
python3 generate_assemble_cf.py
