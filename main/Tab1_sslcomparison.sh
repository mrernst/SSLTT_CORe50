#!/bin/bash 
# 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4 
#SBATCH --time=700:00:00 
#SBATCH --mem=20GB 
#SBATCH --gres=gpu:1 
#SBATCH --partition=your_partition 
#SBATCH --job-name=CORe50Env_ICLR_Tab1
#SBATCH --mail-type=END 
#SBATCH --mail-user=your_email
#SBATCH --output=path/to/the/repository/slurm_%j.out 
#SBATCH --array=0-8%9


# order of array
j=$((SLURM_ARRAY_TASK_ID))

# crossfold k, k in 0,1,2,3,4 is defined by the first argument to the shell script
# e.g. sbatch ablation_experiment.sh 3 runs the 3rd train/test split


contrast_array=(\
'classic' 'time' 'combined' \
'classic' 'time' 'combined' \
'classic' 'time' 'combined'\
)



temp_array=(\
0.1 0.1 0.1 \
0.1 0.1 0.1 \
0.1 0.1 0.1\
)

view_sampling_array=(\
'uniform' 'uniform' 'uniform' \
'uniform' 'uniform' 'uniform' \
'uniform' 'uniform' 'uniform'\
)



loss_array=(\
'SimCLR' 'SimCLR' 'SimCLR' \
'BYOL' 'BYOL' 'BYOL' \
'VICReg' 'VICReg' 'VICReg' \
)


reg_loss_array=(\
'None' 'None' 'None' \
'None' 'None' 'None' \
'None' 'None' 'None'\
)



# no lrate decay for classic augmentations
lratedecay_array=(\
1.0 1.0 1.0 \
1.0 1.0 1.0 \
1.0 1.0 1.0\
)



srun python3 main/train.py \
	--name ICLR_Tab1_$1 \
	--dataset 'CORE50' \
	--n_fix 0.9 \
	--n_fix_per_session 0.5 \
	--main_loss ${loss_array[$j]} \
	--contrast ${contrast_array[$j]} \
	--reg_loss ${reg_loss_array[$j]} \
	--view_sampling ${view_sampling_array[$j]} \
	--save_model \
	--temperature ${temp_array[$j]} \
	--lrate_decay ${lratedecay_array[$j]} \
	--lrate 0.0005 \
	--weight_decay 0.000001 \
	--encoder 'resnet' \
	--projectionhead \
	--n_repeat 1 \
	--test_every 10 \
	--train_split train_alt_$1 \
	--test_split test_alt_$1 \
	--val_split val_alt_$1 \
	