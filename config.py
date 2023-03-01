#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import argparse
import datetime
import torch

# helper function for passing None via CLI
def none_or_str(value):
    if value == 'None':
        return None
    return value

# parse args that correspond to configurations to be experimented on
parser = argparse.ArgumentParser()

parser.add_argument('--name',
                    default='',
                    type=str)
parser.add_argument('--n_fix',
                    default=2,
                    type=float)
parser.add_argument('--n_fix_per_session',
                    default=100,
                    type=float)
parser.add_argument('--n_negative',
                    default=None,
                    type=int)
parser.add_argument('--dataset',
                    default='CORE50',
                    choices=['CORE50'], type=str)
parser.add_argument('--train_split',
                    default='train',
                    type=str)
parser.add_argument('--eval_train_split',
                    default=None,
                    type=str)
parser.add_argument('--test_split',
                    default='test',
                    type=str)
parser.add_argument('--val_split',
                    default='val',
                    type=str)
parser.add_argument('--data_root',
                    default='data', type=str)
parser.add_argument('--encoder',
                    default='resnet',
                    choices=['resnet'], type=str)
parser.add_argument('--projectionhead',
                    dest='projectionhead',
                    action='store_true')
parser.add_argument('--no-projectionhead',
                    dest='projectionhead',
                    action='store_false')
parser.set_defaults(projectionhead=True)
parser.add_argument('--exhaustive_test',
                    dest='exhaustive_test',
                    action='store_true')
parser.add_argument('--no-exhaustive_test',
                    dest='exhaustive_test',
                    action='store_false')
parser.set_defaults(exhaustive_test=True)
parser.add_argument('--main_loss',
                    default='SimCLR',
                    choices=['SimCLR', 'BYOL', 'VICReg', 'supervised', 'supervised_representation'], type=str)
parser.add_argument('--contrast',
                    default='time',
                    choices=['time', 'classic', 'combined', 'supervised', 'nocontrast', 'combined_jitter', 'combined_jitterpluscrop', 'combined_jitterplusgrayscale', 'combined_grayscale'], type=str)
parser.add_argument('--reg_loss',
                    default=None,
                    choices=[None], type=none_or_str)
parser.add_argument('--lrate',
                    default=0.0005,
                    type=float)
parser.add_argument('--weight_decay',
                    default=0.000001,
                    type=float)
parser.add_argument('--training_percentage',
                    default=1.0,
                    type=float)
parser.add_argument('--testing_percentage',
                    default=0.1,
                    type=float)
parser.add_argument('--cosine_decay',
                    dest='cosine_decay',
                    action='store_true')
parser.set_defaults(cosine_decay=False)
parser.add_argument('--exp_decay',
                    dest='exp_decay',
                    action='store_true')
parser.set_defaults(exp_decay=False)
parser.add_argument('--lrate_decay',
                    default=1.0,
                    type=float)
parser.add_argument('--decorr_weight',
                    default=0.4,
                    type=float)
parser.add_argument('--temperature',
                    default=0.1,
                    type=float)
parser.add_argument('--similarity',
                    default='cosine',
                    choices=['cosine', 'RBF'], type=str)
parser.add_argument('--view_sampling',
                    default='randomwalk',
                    choices=['randomwalk', 'uniform'], type=str)
parser.add_argument('--shuffle_objects',
                    dest='shuffle_object_order',
                    action='store_true')
parser.add_argument('--no-shuffle_objects',
                    dest='shuffle_object_order',
                    action='store_false')
parser.set_defaults(shuffle_object_order=True)
parser.add_argument("--seed",type=int,default=0)

parser.add_argument('--save_model',
                    dest='save_model',
                    action='store_true')
parser.add_argument('--no-save_model',
                    dest='save_model',
                    action='store_false')
parser.set_defaults(save_model=True)

parser.add_argument('--save_embedding',
                    default=False,
                    type=bool)



parser.add_argument('--feature_dim',
                    default=128,
                    type=int)
parser.add_argument('--hidden_dim',
                    default=256,
                    type=int)
parser.add_argument('--log_dir',
                    default='save',
                    type=str)
parser.add_argument('--n_epochs',
                    default=100,
                    type=int)
parser.add_argument('--n_repeat',
                    default=5,
                    type=int)
parser.add_argument('--test_every',
                    default=1,
                    type=int)
parser.add_argument('--save_every',
                    default=100,
                    type=int)
parser.add_argument('--tau',
                    default=0.996,
                    type=float)
parser.add_argument('--batch_size',
                    default=256,
                    type=int)
parser.add_argument('--crop_size',
                    default=128,
                    type=int)


# VICREG arguments
parser.add_argument("--sim-coeff", type=float, default=25.0,
                                    help='Invariance regularization loss coefficient')
parser.add_argument("--std-coeff", type=float, default=25.0,
                    help='Variance regularization loss coefficient')
parser.add_argument("--cov-coeff", type=float, default=1.0,
                    help='Covariance regularization loss coefficient')



parser.add_argument('--knn_batch_size',
                    default=256,
                    type=int)
parser.add_argument("--experiment_dir", help="full path to experiment directory for loading files",
                type=str)

args = parser.parse_args()

N_fix = args.n_fix
N_fix_per_session = args.n_fix_per_session
N_negative = args.n_negative
DATASET = args.dataset
TRAIN_SPLIT = args.train_split
EVAL_TRAIN_SPLIT = args.eval_train_split
TEST_SPLIT = args.test_split
VAL_SPLIT = args.val_split
DATA_ROOT = args.data_root
ENCODER = args.encoder
PROJECTIONHEAD = args.projectionhead
CONTRAST = args.contrast
MAIN_LOSS = args.main_loss
REG_LOSS = args.reg_loss
SIMILARITY = args.similarity
SHUFFLE_OBJECT_ORDER = args.shuffle_object_order
VIEW_SAMPLING = args.view_sampling
SAVE_MODEL = args.save_model
SAVE_EMBEDDING = args.save_embedding
LRATE = args.lrate
WEIGHT_DECAY = args.weight_decay
LR_DECAY_RATE = args.lrate_decay
DECORR_WEIGHT = args.decorr_weight
COSINE_DECAY = args.cosine_decay
EXP_DECAY = args.exp_decay
TEMPERATURE = args.temperature
DATASET = args.dataset
SEED = args.seed
RUN_NAME = f'{datetime.datetime.now().strftime("%d-%m-%y_%H:%M")}_{args.name}_seed_{SEED}_{DATASET}_aug_{CONTRAST}_{VIEW_SAMPLING}_{MAIN_LOSS}_reg_{REG_LOSS}_nfix_{N_fix}_persess_{N_fix_per_session}'


# only implemented on CORe50
TRAINING_PERCENTAGE = args.training_percentage
TESTING_PERCENTAGE = args.testing_percentage
EXHAUSTIVE_TEST = args.exhaustive_test
KNN_BATCH_SIZE = args.knn_batch_size

FEATURE_DIM = args.feature_dim
HIDDEN_DIM = args.hidden_dim
LOG_DIR = args.log_dir
N_EPOCHS = args.n_epochs
N_REPEAT = args.n_repeat
TEST_EVERY = args.test_every
SAVE_EVERY = args.save_every
TAU = args.tau
BATCH_SIZE = args.batch_size
CROP_SIZE = args.crop_size

# only vicreg parameters
VICREG_SIM_COEFF = args.sim_coeff
VICREG_STD_COEFF = args.std_coeff
VICREG_COV_COEFF = args.cov_coeff

# configurations that are not tuned
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PRIOR = 'gaussian'
LR_DECAY_EPOCHS = [0] #[700, 800, 900]


# _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
