#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=secondary_concrete_vae_hyperbanding
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=/users/k1754828/ml_stuttering_project/core_analysis/preprocessing/preprocessing_phase_two/primary_concrete_feature_selection/con_vae_selector_control_file/con_vae_selector_control_file-%j.output
#SBATCH --error=/users/k1754828/ml_stuttering_project/core_analysis/preprocessing/preprocessing_phase_two/primary_concrete_feature_selection/con_vae_selector_control_file/con_vae_selector_control_file-%j.error

wd=/users/k1754828/ml_stuttering_project/core_analysis/preprocessing/preprocessing_phase_two/primary_concrete_feature_selection/con_vae_selector_control_file


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main.py
