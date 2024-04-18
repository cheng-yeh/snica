#!/bin/bash
#SBATCH -J main_real                            # Job name
#SBATCH --account=gts-rs275                     # charge account
#SBATCH -N 1 --ntasks-per-node=1                # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=32G                       # Memory per core
#SBATCH -t 600                                   # Duration of the job (Ex: 15 mins)
#SBATCH -q inferno                              # QOS Name
#SBATCH -o Report-%j.out                        # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=cchen847@gatech.edu         # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory

module load anaconda3/2022.05                   # Load module dependencies
srun conda create -n snica python=3.10 -y
srun conda activate snica
srun pip install --upgrade pip
srun pip install torch==1.13.1 torchvision torchaudio
srun pip install -r requirements.txt
srun pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python main_real.py --num-epoch 10000 -t 73305 -n 5 -d 5 -k 2 --hidden-units-enc 32 --hidden-units-dec 32 --input-data data/alice_eeg/24chans_fmin4000_fmax30000_rwin2_samp5.pkl
