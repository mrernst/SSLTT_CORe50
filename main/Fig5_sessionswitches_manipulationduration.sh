#!/bin/bash 
# 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4 
#SBATCH --time=700:00:00 
#SBATCH --mem=20GB 
#SBATCH --gres=gpu:1 
#SBATCH --partition=your_partition 
#SBATCH --job-name=CORe50Env_ICLR_Fig5
#SBATCH --mail-type=END 
#SBATCH --mail-user=your_email
#SBATCH --output=path/to/the/repository/slurm_%j.out 
#SBATCH --array=0-24%25


# order of array
j=$((SLURM_ARRAY_TASK_ID))


# crossfold k, k in 0,1,2,3,4 is defined by the first argument to the shell script
# e.g. sbatch ablation_experiment.sh 3 runs the 3rd train/test split

n_fix_array=(\
0.5 0.8 0.9 0.95 0.98 \
0.5 0.8 0.9 0.95 \
0.5 0.8 0.9 \
0.5 0.8 \
0.5 \
0.8 0.9 0.95 0.98 \
0.9 0.95 0.98 \
0.95 0.98 \
0.98\
)


n_fix_per_session_array=(\
0.5 0.8 0.9 0.95 0.98 \
0.8 0.9 0.95 0.98 \
0.9 0.95 0.98 \
0.95 0.98 \
0.98 \
0.5 0.8 0.9 0.95 \
0.5 0.8 0.9 \
0.5 0.8 \
0.5\
)


srun python3 main/train.py \
	--name ICLR_Fig5_$1 \
	--dataset 'CORE50' \
	--n_fix ${n_fix_array[$j]} \
	--n_fix_per_session ${n_fix_per_session_array[$j]} \
	--main_loss 'SimCLR' \
	--contrast 'time' \
	--view_sampling 'randomwalk' \
	--temperature 0.1 \
	--lrate_decay 1.0 \
	--lrate 0.0005 \
	--weight_decay 0.000001 \
	--encoder 'resnet' \
	--projectionhead \
	--n_repeat 1 \
	--test_every 10 \
	--train_split train_alt_$1 \
	--test_split test_alt_$1 \
	--val_split val_alt_$1 \



# --- end of experiment ---
