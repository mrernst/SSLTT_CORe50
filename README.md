# Time to augment visual self-supervised learning
## The CORe50 Environment

<p align="center">
  <img src="https://github.com/mrernst/timetoaugment/blob/main/img/core50_procedure.jpg" width="1000">

This code accompanies the ICLR paper "Time to augment visual self-supervised learning". It contains all the necessary building blocks to replicate the results reported.


## Getting started with the code

* Clone the repository from here (for now it is just a folder in the supplement)
* Make sure you have all the dependencies installed, see also requirements.txt
* Download the corresponding CORe50 dataset files
* Start an experiment on your local machine with python3 main/train.py

### Prerequisites

* [numpy](http://www.numpy.org/)
* [pytorch](https://www.pytorch.org/)
* [matplotlib](https://matplotlib.org/)
* [tensorboard](https://tensorflow.org/)
* [pandas](https://pandas.pydata.org)
* [tqdm](https://pypi.org/project/tqdm/)




### Repository structure

```bash
.
├── codetemplate                       # Template for collaboration purposes
├── config.py                          # Configuration for experiment parameters
├── data                               
│   └── core50_128x128                 # CORe50 dataset (not included, please download)
├── img
│   └── core50_procedure_.png          # Image that displays in README.md
├── LICENSE                            # MIT License
├── main
│   ├── Fig4_baselines.sh              # slurm script to send experiment to the cluster
│   ├── Fig4_combined.sh               # slurm script to send experiment to the cluster
│   ├── Fig4_ablation_experiment.sh    # slurm script to send experiment to the cluster
│   ├── Fig7_ablation_experiment.sh    # slurm script to send experiment to the cluster
│   └── train.py                       # main training file     		                 		    
├── README.md                          # ReadMe File
├── requirements.txt                   # conda/pip requirements
├── utils
│   ├── augmentations.py               # augmentations for standard SimCLR
│   ├── datasets.py                    # dataloading and sampling
│   ├── evaluation.py                  # evaluation methods and analytics
│   ├── general.py                     # general IO utilities
│   ├── losses.py                      # definition of loss functions
└── └── networks.py                    # network definition, e.g. ResNet

```

### Installation guide

#### Forking the repository

Fork a copy of this repository onto your own GitHub account and `clone` your fork of the repository into your computer, inside your favorite folder, using:

`git clone "PATH_TO_FORKED_REPOSITORY"`

#### Setting up the environment

Install [Python 3.9](https://www.python.org/downloads/release/python-395/) and the [conda package manager](https://conda.io/miniconda.html) (use miniconda). Navigate to the project directory inside a terminal and create a virtual environment (replace <ENVIRONMENT_NAME>, for example, with `CORe50Env`) and install the [required packages](requirements.txt):

`conda create -n <ENVIRONMENT_NAME> --file requirements.txt python=3.9`

Activate the virtual environment:

`source activate <ENVIRONMENT_NAME>`


#### Downloading the dataset
The underlying CORe50 dataset (Lomonaco and Maltoni, 2017) is publicly available [here](http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip).
Please download and store at your location of choice (default would be at ./data as indicated in the repository structure)


#### Starting an experiment
Starting an experiment is straight forward. Just execute the script from the main level and specify your options via the command line.

##### Example A) A run with the CORe50Env
```bash
python3 main/train.py \
	--name CEnv_Exp_0 \                             # specify experiment name
	--data_root './data/' \                         # specify where you put the CORe50 dataset
	--n_fix 0.95 \                                  # specify N_o as float probability [0,1]
	--n_fix_per_session 0.95 \                      # specify N_s as float probability [0,1]
	--contrast 'time' \                             # choose 'time' or 'combined'
	--view_sampling randomwalk \                    # choose 'randomwalk' or 'uniform'
	--test_every 10 \                               # test every 10 epochs
	--train_split train_alt_0 \                     # choose the splits for cross-validation (k in range(5))
	--test_split test_alt_0 \
	--val_split val_alt_0 \

```

##### Example B) A comparison run with standard SimCLR
```bash
python3 main/train.py \
	--name SimCLR_Exp_0 \                           # specify experiment name
	--data_root './data/' \                         # specify where you put the CORe50 dataset
	--contrast 'classic' \                          # choose 'classic' for SimCLR type contrasts
	--test_every 10 \                               # test every 10 epochs
	--train_split train_alt_0 \                     # choose the splits for cross-validation (k in range(5))
	--test_split test_alt_0 \
	--val_split val_alt_0 \

```

##### Example C) A comparison run with a supervised model
```bash
python3 main/train.py \
	--name Supervised_Exp_0 \                       # specify experiment name
	--data_root './data/' \                         # specify where you put the CORe50 dataset
	--contrast 'nocontrast' \                       # choose 'nocontrast' for supervised experiments
	--main_loss 'supervised' \                      # supervised loss
	--test_every 10 \                               # test every 10 epochs
	--train_split train_alt_0 \                     # choose the splits for cross-validation (k in range(5))
	--test_split test_alt_0 \
	--val_split val_alt_0 \

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
