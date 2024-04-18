#!/bin/bash
#SBATCH -J custom_dataset_train                 # Job name
#SBATCH --account=gts-rs275                     # charge account
#SBATCH -N 1 --gres=gpu:RTX_6000:2              # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=24G                       # Memory per core
#SBATCH -t 1440                                 # Duration of the job (Ex: 15 mins)
#SBATCH -q inferno                              # QOS Name
#SBATCH -o Report-%j.out                        # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=cchen847@gatech.edu         # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory

module load anaconda3/2022.05                   # Load module dependencies
python3 -m venv bams_env
source bams_env/bin/activate
srun pip install --upgrade pip
srun pip install -e .
python3 custom_dataset.py --job train --input_train train_24chans_fmin10_fmax25000_rwin40_samp20.pkl --input_submission test_24chans_fmin10_fmax25000_rwin40_samp20.pkl
