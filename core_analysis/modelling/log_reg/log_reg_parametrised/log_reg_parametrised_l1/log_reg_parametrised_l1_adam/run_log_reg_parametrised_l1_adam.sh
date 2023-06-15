#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=log_reg_net_params
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=$log/core_analysis/modelling/log_reg/log_reg_parametrised/log_reg_parametrised_l1/log_reg_parametrised_l1_adam/log_reg_parametrised_l1_adam-%j.output
#SBATCH --error=$log/core_analysis/modelling/log_reg/log_reg_parametrised/log_reg_parametrised_l1/log_reg_parametrised_l1_adam/log_reg_parametrised_l1_adam-%j.error

wd=$modelling/ocre_analysis/modelling/log_reg/log_reg_parametrised/log_reg_parametrised_l1/log_reg_parametrised_l1_adam


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main_reopen.py

