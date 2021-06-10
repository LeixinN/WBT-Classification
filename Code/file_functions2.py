# Author: Leixin NIE

# Import packages section
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
# Import functions
from file_functions import roc_postprocess, train_model, evaluate_model


# Train and evaluate model using 5-fold cross validation
def train_evaluate_model_CV(deep_net_input, dataloaders_input, dataset_sizes, device, model_name, fold=5, num_epochs=200, epoch_stride=20, stop_early=True):
    # Determine the value of flag_mixup
    if len(iter(dataloaders_input[0]['train']).next()[1].shape)-1:
        flag_mixup = True
    else:
        flag_mixup = False

    # 5-fold cross validation
    deep_nets = []
    index_training = []
    index_evaluating = []
    cf_matrix = []
    for ii in range(fold):
        print('------------Fold-{:.0f}------------'.format(ii+1))
        print()

        deep_net = copy.deepcopy(deep_net_input[ii])
        dataloaders = dataloaders_input[ii]
        
        deep_net = deep_net.to(device)
        # Set up criterion
        if flag_mixup:
            criterion = {
                'train': SoftCrossEntropyLoss(nn.CrossEntropyLoss(reduction='none')),
                'test': nn.CrossEntropyLoss()
            }
        else:
            criterion = {
                'train': nn.CrossEntropyLoss(),
                'test': nn.CrossEntropyLoss()
            }
        # Set up optimizer & scheduler
        optimizer = optim.Adam(deep_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
        # Train a model
        deep_net, loss_all, acc_all = train_model(deep_net, criterion, optimizer, exp_lr_scheduler, num_epochs, dataloaders, dataset_sizes, device, model_name, epoch_stride, flag_mixup, current_fold=ii, stop_early=stop_early)
        # Evaluate the model
        auc_score, fpr, tpr, cf_mat = evaluate_model(deep_net, dataloaders, device)

        deep_nets.append(deep_net)
        index_training.append({'loss':loss_all, 'accurate':acc_all})
        index_evaluating.append({'auc':auc_score, 'fpr':fpr, 'tpr':tpr})
        cf_matrix.append(cf_mat)

    # Postprocess
    index_evaluating_processed = roc_postprocess(index_evaluating)

    return index_evaluating_processed, cf_matrix, deep_nets, index_training, index_evaluating


