#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import os
import sys

from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms

# configuration module
# -----

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config

# custom libraries
# -----
from utils.datasets import CORE50Dataset
from utils.networks import ResNet18, MLPHead
from utils.losses import SimCLR_TT_Loss, BYOL_TT_Loss, VICReg_TT_Loss
from utils.general import update_target_network_parameters, initialize_target_network
from utils.general import save_model, load_model, save_args, mkdir_p
from utils.evaluation import get_representations, lls_fit, lls_eval, supervised_eval, wcss_bcss
from utils.augmentations import get_transformations, TwoContrastTransform


# similarity functions dictionary
SIMILARITY_FUNCTIONS = {
    'cosine': lambda x, x_pair: F.cosine_similarity(x.unsqueeze(1), x_pair.unsqueeze(0), dim=2),
    'RBF': lambda x, x_pair: -torch.cdist(x, x_pair)
}

# loss dictionary for different losses
MAIN_LOSS = {
    'SimCLR': SimCLR_TT_Loss(SIMILARITY_FUNCTIONS[config.SIMILARITY], config.BATCH_SIZE, config.TEMPERATURE),
    'BYOL': BYOL_TT_Loss(SIMILARITY_FUNCTIONS[config.SIMILARITY]),
    'VICReg': VICReg_TT_Loss(),
    'supervised': lambda x, x_pair, labels: F.cross_entropy(x, labels),
    'supervised_representation': lambda x, x_pair, labels: F.cross_entropy(x, labels),
}


REG_LOSS = {}


DATASETS = {
    'CORE50': {'class': CORE50Dataset,
            'size': int(107953*config.TRAINING_PERCENTAGE) if config.TRAIN_SPLIT=='train' else int(118750*config.TRAINING_PERCENTAGE),
            'rgb_mean': (0.6000, 0.5681, 0.5411),
            'rgb_std': (0.1817, 0.1944, 0.2039),
    },
}

# define manual random seed
torch.manual_seed(config.SEED)
random.seed(config.SEED)
np.random.seed(config.SEED)


# custom function
# -----


def train():
    path_to_experiment = os.path.join(config.LOG_DIR, config.RUN_NAME)
    # make directory
    mkdir_p(path_to_experiment)
    # write config to a json file
    save_args(path_to_experiment, config.args.__dict__)
    
    # prepare tensorboard writer
    writer = SummaryWriter(log_dir=path_to_experiment)
    
    # get transformations for validation and for training
    train_transform, val_transform = get_transformations(
        contrast_type=config.CONTRAST,
        rgb_mean=DATASETS[config.DATASET]['rgb_mean'],
        rgb_std=DATASETS[config.DATASET]['rgb_std'],
        crop_size=config.CROP_SIZE)
    
    dataset_train = DATASETS[config.DATASET]['class'](
        root=config.DATA_ROOT,
        split=config.TRAIN_SPLIT,
        transform=train_transform,
        contrastive=True if (config.CONTRAST == 'time' or \
                    config.CONTRAST == 'combined') else False,
        sampling_mode=config.VIEW_SAMPLING,
        shuffle_object_order=config.SHUFFLE_OBJECT_ORDER,
        buffer_size=DATASETS[config.DATASET]['size'],
        n_fix=config.N_fix)
    
    dataloader_train = DataLoader(dataset_train, batch_size=config.BATCH_SIZE,
                                  num_workers=4, shuffle=True, drop_last=True)
    
    dataset_train_eval = DATASETS[config.DATASET]['class'](
        root=config.DATA_ROOT,
        # choose the evaluation training split, if not defined use the training split
        split=config.EVAL_TRAIN_SPLIT if config.EVAL_TRAIN_SPLIT else config.TRAIN_SPLIT,
        transform=val_transform,
        contrastive=False,
        sampling_mode=config.VIEW_SAMPLING,
        shuffle_object_order=config.SHUFFLE_OBJECT_ORDER,
        buffer_size=DATASETS[config.DATASET]['size'],
        n_fix=config.N_fix)
    dataloader_train_eval = DataLoader(dataset_train_eval, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=False)
    dataset_test = DATASETS[config.DATASET]['class'](
        root=config.DATA_ROOT,
        split=config.TEST_SPLIT,
        transform=val_transform,
        contrastive=False,
        sampling_mode=config.VIEW_SAMPLING,
        shuffle_object_order=config.SHUFFLE_OBJECT_ORDER,
        buffer_size= DATASETS[config.DATASET]['size'],
        n_fix=config.N_fix)
    dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=False)
        
    # load the validation set
    dataset_val = DATASETS[config.DATASET]['class'](
        root=config.DATA_ROOT,
        split=config.VAL_SPLIT,
        transform=val_transform,
        contrastive=False,
        sampling_mode=config.VIEW_SAMPLING,
        shuffle_object_order=config.SHUFFLE_OBJECT_ORDER,
        buffer_size= DATASETS[config.DATASET]['size'],
        n_fix=config.N_fix)
    dataloader_val = DataLoader(dataset_val, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=False)
    
    # reset testing sets to a smaller percentage of the data
    # to save evaluation time on the cluster during intermediate testing
    dataset_train_eval.dataset_percentage = config.TESTING_PERCENTAGE
    dataset_test.dataset_percentage = config.TESTING_PERCENTAGE
    dataset_val.dataset_percentage = config.TESTING_PERCENTAGE
        
    if config.ENCODER == 'resnet':
        net = ResNet18(no_classes=dataset_train.n_classes).to(config.DEVICE)
        net_target = ResNet18(no_classes=dataset_train.n_classes).to(config.DEVICE).eval()
    else:
        raise NotImplementedError('[INFO] Specified Encoder is not implemented')
    
    
    # specific when BYOL-TT is used
    if config.MAIN_LOSS == 'BYOL':
        # net_target.train()
        predictor = MLPHead(config.FEATURE_DIM, config.HIDDEN_DIM, config.FEATURE_DIM).to(config.DEVICE)
        # initialize target network
        initialize_target_network(net_target, net)
    
    
        optimizer = torch.optim.AdamW(list(net.parameters()) + list(predictor.parameters()), lr=config.LRATE, weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.LRATE, weight_decay=config.WEIGHT_DECAY)
    
    
    # get initial result and save plot and record
    if config.MAIN_LOSS == 'supervised':
        
        # Evaluate on test set
        # -----
        acc, test_loss, class_cm = supervised_eval(net, dataloader_test, F.cross_entropy, dataset_test.n_classes)
        writer.add_scalar('accloss/test/class/accuracy', acc, 0)
        writer.add_scalar('accloss/test/class/loss', test_loss, 0)
        class_cm.to_tensorboard(writer, dataset_test.labels, 0, label='test/class/cm',)
        
        
        # Evaluate on validation set
        # -----
        acc, test_loss, class_cm = supervised_eval(net, dataloader_val, F.cross_entropy, dataset_val.n_classes)
        writer.add_scalar('accloss/val/class/accuracy', acc, 0)
        writer.add_scalar('accloss/val/class/loss', test_loss, 0)
        class_cm.to_tensorboard(writer, dataset_test.labels, 0, label='val/class/cm',)
    else:

        features_train_eval, labels_train_eval = get_representations(net, dataloader_train_eval)
        
        # Evaluate on test set
        # -----
        features_test, labels_test = get_representations(net, dataloader_test)
        
        lstsq_model = lls_fit(features_train_eval, labels_train_eval, dataset_train_eval.n_classes)
        pred, acc = lls_eval(lstsq_model, features_test, labels_test)
    
        wb = wcss_bcss(features_test, labels_test, dataset_test.n_classes)
        
        print(f'Initial result: Read-Out Acc:{acc * 100:>6.2f}%, WCSS/BCSS:{wb:>8.4f}')
        writer.add_scalar('accloss/test/class/accuracy', acc, 0)
        writer.add_scalar('analytics/test/class/WCSS-BCSS', wb, 0)
        
        
        # Evaluate on validation set
        # -----
        features_val, labels_val = get_representations(net, dataloader_val)
        pred, acc = lls_eval(lstsq_model, features_val, labels_val)
        wb = wcss_bcss(features_val, labels_val, dataset_val.n_classes)
        
        writer.add_scalar('accloss/val/class/accuracy', acc, 0)
        writer.add_scalar('analytics/val/class/WCSS-BCSS', wb, 0)
        
        
        if config.EXHAUSTIVE_TEST:
            # Relabelling the data
            # -----
            # change labelling of evaluation set
            # dataloader_train_eval loads dataset_train_eval
            dataset_train_eval.label_by = 'object'
            # dataloader_test loads dataset_test
            dataset_test.label_by = 'object'
            # dataloader_val loads dataset_val
            dataset_val.label_by = 'object'
            # refit lls
            features_train_eval, labels_train_eval = get_representations(net, dataloader_train_eval)
            lstsq_model = lls_fit(features_train_eval, labels_train_eval, dataset_train_eval.n_classes)
            
            
            # Evaluate on test set
            # -----
            features_test, labels_test = get_representations(net, dataloader_test)
            pred, acc = lls_eval(lstsq_model, features_test, labels_test)
            wb = wcss_bcss(features_test, labels_test, dataset_test.n_classes)
            writer.add_scalar('accloss/test/object/accuracy', acc, 0)
            writer.add_scalar('analytics/test/object/WCSS-BCSS', wb, 0)
            
            
            # Evaluate on validation set
            # -----
            features_val, labels_val = get_representations(net, dataloader_val)
            pred, acc = lls_eval(lstsq_model, features_val, labels_val)
            wb = wcss_bcss(features_val, labels_val, dataset_val.n_classes)
            writer.add_scalar('accloss/val/object/accuracy', acc, 0)
            writer.add_scalar('analytics/val/object/WCSS-BCSS', wb, 0)
            
            
            # Relabelling the data
            # -----
            # change labelling of evaluation set
            # dataloader_train_eval loads dataset_train_eval
            dataset_train_eval.label_by = 'session'
            # dataloader_test loads dataset_test
            dataset_test.label_by = 'session'
            # dataloader_val loads dataset_val
            dataset_val.label_by = 'session'
            # refit lls
            features_train_eval, labels_train_eval = get_representations(net, dataloader_train_eval)
            lstsq_model = lls_fit(features_train_eval, labels_train_eval, dataset_train_eval.n_classes)
            
            # Evaluate on test set
            # -----
            features_test, labels_test = get_representations(net, dataloader_test)
            pred, acc = lls_eval(lstsq_model, features_test, labels_test)
            wb = wcss_bcss(features_test, labels_test, dataset_test.n_classes)
            writer.add_scalar('accloss/test/session/accuracy', acc, 0)
            writer.add_scalar('analytics/test/session/WCSS-BCSS', wb, 0)
            
            # Evaluate on validation set
            # -----
            features_val, labels_val = get_representations(net, dataloader_val)
            pred, acc = lls_eval(lstsq_model, features_val, labels_val)
            wb = wcss_bcss(features_val, labels_val, dataset_val.n_classes)
            writer.add_scalar('accloss/val/session/accuracy', acc, 0)
            writer.add_scalar('analytics/val/session/WCSS-BCSS', wb, 0)
            
            # Relabelling the data
            # -----
            # revert to usual class labelling
            dataset_val.label_by = 'class'
            dataset_train_eval.label_by = 'class'
            dataset_test.label_by = 'class'
            
        if config.SAVE_EMBEDDING:
            writer.add_embedding(features_test, tag='Embedding', global_step=0)
    
    
    # learning rate scheduling
    if config.COSINE_DECAY:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.N_EPOCHS,
            eta_min=config.LRATE * (config.LR_DECAY_RATE ** 3))
    elif config.EXP_DECAY:
        scheduler = ExponentialLR(optimizer, 1.0)
    else:
        scheduler =  StepLR(optimizer, 10, config.LR_DECAY_RATE)
    
    
    # -----------------
    # Epoch loop
    # -----------------
    
    epoch_loop = tqdm(range(config.N_EPOCHS), ncols=80)
    for epoch in epoch_loop:
        epoch_loop.set_description(f"Method: {config.RUN_NAME.split('~')[0]}, Epoch: {epoch + 1}")
        # buffer refresh
        if config.CONTRAST != 'classic':
            dataset_train.refresh_buffer()
        
        net.train()
        training_loop = tqdm(dataloader_train)
        for (x, x_pair), labels in training_loop:
            x, y = torch.cat([x, x_pair], 0).to(config.DEVICE), labels.to(config.DEVICE)
            representation, projection = net(x)
            projection, pair = projection.split(projection.shape[0]//2)
            if config.MAIN_LOSS == 'BYOL':
                # compute query feature
                predictions_from_view_1 = predictor(projection)
                predictions_from_view_2 = predictor(pair)
                # compute key feature
                with torch.no_grad():
                    _ , target_output = net_target(x)
                    targets_to_view_2, targets_to_view_1 = target_output.split(target_output.shape[0]//2)
                
                loss = MAIN_LOSS[config.MAIN_LOSS](predictions_from_view_1, targets_to_view_1, y)
                loss += MAIN_LOSS[config.MAIN_LOSS](predictions_from_view_2, targets_to_view_2, y)
                loss = loss.mean()
                update_target_network_parameters(net_target, net, config.TAU)
            else:
                loss = MAIN_LOSS[config.MAIN_LOSS](projection, pair, y)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            training_loop.set_description(f'Loss: {loss.item():>8.4f}')
        
        # update learning rate
        lr_decay_steps = torch.sum(epoch > torch.Tensor(config.LR_DECAY_EPOCHS))
        if (lr_decay_steps > 0 and not(config.COSINE_DECAY or config.EXP_DECAY)):
            scheduler.gamma = config.LR_DECAY_RATE ** lr_decay_steps
        scheduler.step()
        
        if config.SAVE_MODEL:
            if (epoch + 1) % config.SAVE_EVERY == 0 or (epoch+1) == config.N_EPOCHS:
                save_model(net, writer, epoch + 1)
        
        if (epoch+1) % config.TEST_EVERY == 0 or (epoch+1) == config.N_EPOCHS:
            # set network to evaluation mode
            net.eval()
            
            if (epoch+1) == config.N_EPOCHS:
                # reset testing sets to 100% of the data at final evaluation
                dataset_train_eval.dataset_percentage = config.TRAINING_PERCENTAGE
                dataset_test.dataset_percentage = config.TRAINING_PERCENTAGE
                dataset_val.dataset_percentage = config.TRAINING_PERCENTAGE            
            
            if config.MAIN_LOSS == 'supervised':
                writer.add_scalar('accloss/train/loss', loss.item(), epoch + 1)
                writer.add_scalar('analytics/learningrate', scheduler.get_last_lr()[0], epoch + 1)

                acc, test_loss, class_cm = supervised_eval(net, dataloader_test, F.cross_entropy, dataset_test.n_classes)
                writer.add_scalar('accloss/test/class/accuracy', acc, epoch + 1)
                writer.add_scalar('accloss/test/class/loss', test_loss, epoch + 1)
                
                acc, test_loss, class_cm = supervised_eval(net, dataloader_val, F.cross_entropy, dataset_val.n_classes)
                writer.add_scalar('accloss/val/class/accuracy', acc, epoch + 1)
                writer.add_scalar('accloss/val/class/loss', test_loss, epoch + 1)
            else:
                features_train_eval, labels_train_eval = get_representations(net, dataloader_train_eval)
                features_test, labels_test = get_representations(net, dataloader_test)
                lstsq_model = lls_fit(features_train_eval, labels_train_eval, dataset_train_eval.n_classes)
                pred, acc = lls_eval(lstsq_model, features_test, labels_test)
                wb = wcss_bcss(features_test, labels_test, dataset_test.n_classes)
                print(f"Method: {config.RUN_NAME.split('~')[0]}, Epoch: {epoch + 1}, "
                      f"Read-Out Acc:{acc * 100:>6.2f}%, WCSS/BCSS:{wb:>8.4f}")
        
                # record results
                writer.add_scalar('accloss/train/loss', loss.item(), epoch + 1)
                writer.add_scalar('accloss/test/class/accuracy', acc, epoch + 1)
                writer.add_scalar('analytics/test/class/WCSS-BCSS', wb, epoch + 1)
                writer.add_scalar('analytics/learningrate', scheduler.get_last_lr()[0], epoch + 1)
                
                
                features_val, labels_val = get_representations(net, dataloader_val)
                pred, acc = lls_eval(lstsq_model, features_val, labels_val)
                wb = wcss_bcss(features_val, labels_val, dataset_val.n_classes)
                writer.add_scalar('accloss/val/class/accuracy', acc, epoch + 1)
                writer.add_scalar('analytics/val/class/WCSS-BCSS', wb, epoch + 1)
                
                if config.EXHAUSTIVE_TEST:
                    # Relabelling the data
                    # -----
                    # change labelling of evaluation set
                    # dataloader_train_eval loads dataset_train_eval
                    dataset_train_eval.label_by = 'object'
                    # dataloader_test loads dataset_test
                    dataset_test.label_by = 'object'
                    # dataloader_val loads dataset_val
                    dataset_val.label_by = 'object'
                    # refit lls
                    features_train_eval, labels_train_eval = get_representations(net, dataloader_train_eval)
                    lstsq_model = lls_fit(features_train_eval, labels_train_eval, dataset_train_eval.n_classes)
                    
                    
                    
                    # Evaluate on test set
                    # -----
                    features_test, labels_test = get_representations(net, dataloader_test)
                    pred, acc = lls_eval(lstsq_model, features_test, labels_test)
                    wb = wcss_bcss(features_test, labels_test, dataset_test.n_classes)
                    writer.add_scalar('accloss/test/object/accuracy', acc, epoch + 1)
                    writer.add_scalar('analytics/test/object/WCSS-BCSS', wb, epoch + 1)
                    
                    
                    # Evaluate on validation set
                    # -----
                    features_val, labels_val = get_representations(net, dataloader_val)
                    pred, acc = lls_eval(lstsq_model, features_val, labels_val)
                    wb = wcss_bcss(features_val, labels_val, dataset_val.n_classes)
                    writer.add_scalar('accloss/val/object/accuracy', acc, epoch + 1)
                    writer.add_scalar('analytics/val/object/WCSS-BCSS', wb, epoch + 1)
                    
                    
                    # Relabelling the data
                    # -----
                    # change labelling of evaluation set
                    # dataloader_train_eval loads dataset_train_eval
                    dataset_train_eval.label_by = 'session'
                    # dataloader_test loads dataset_test
                    dataset_test.label_by = 'session'
                    # dataloader_val loads dataset_val
                    dataset_val.label_by = 'session'
                    # refit lls
                    features_train_eval, labels_train_eval = get_representations(net, dataloader_train_eval)
                    lstsq_model = lls_fit(features_train_eval, labels_train_eval, dataset_train_eval.n_classes)
                    
                    # Evaluate on test set
                    # -----                    
                    features_test, labels_test = get_representations(net, dataloader_test)
                    pred, acc = lls_eval(lstsq_model, features_test, labels_test)
                    wb = wcss_bcss(features_test, labels_test, dataset_test.n_classes)
                    writer.add_scalar('accloss/test/session/accuracy', acc, epoch + 1)
                    writer.add_scalar('analytics/test/session/WCSS-BCSS', wb, epoch + 1)
                    
                    # Evaluate on validation set
                    # -----
                    features_val, labels_val = get_representations(net, dataloader_val)
                    pred, acc = lls_eval(lstsq_model, features_val, labels_val)
                    wb = wcss_bcss(features_val, labels_val, dataset_val.n_classes)
                    writer.add_scalar('accloss/val/session/accuracy', acc, epoch + 1)
                    writer.add_scalar('analytics/val/session/WCSS-BCSS', wb, epoch + 1)
                    
                    
                    # revert to usual labelling
                    dataset_val.label_by = 'class'
                    dataset_train_eval.label_by = 'class'
                    dataset_test.label_by = 'class'
            
            
            # set network back to training mode
            net.train()
            
            if config.SAVE_EMBEDDING:
                writer.add_embedding(features_test, tag='Embedding', global_step=epoch + 1)
            


# ----------------
# main program
# ----------------

if __name__ == '__main__':
    for i in range(config.N_REPEAT):
        config.RUN_NAME = config.RUN_NAME.rsplit('~')[0] + f'~{i}'
        train()

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
