# Deep Transfer Learning for Wideband Tympanometry 2D-Classification
# Function Files
# Author: Leixin NIE


# Import packages section
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sip
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import cv2
import GlobalVar
from sklearn.model_selection import StratifiedKFold
from torch.utils import data
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from scipy.special import gamma
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D


# Define section
def data_preprocess_JM18(data_preprocess):
    "Preprocessing the struct data, and transform each object to a <14x1> vector"
    freq_centralized = np.array([250, 500, 1000, 2000, 4000, 8000]) # 1-octave band
    freq_bound = freq_centralized * np.sqrt(2)
    freq = data_preprocess[0, 0]['Frequency']
    freq = freq[0, :]
    freq_band0_bool = np.logical_and(freq <= freq_bound[0], freq > 0)
    freq_band1_bool = np.logical_and(freq <= freq_bound[1], freq > freq_bound[0])
    freq_band2_bool = np.logical_and(freq <= freq_bound[2], freq > freq_bound[1])
    freq_band3_bool = np.logical_and(freq <= freq_bound[3], freq > freq_bound[2])
    freq_band4_bool = np.logical_and(freq <= freq_bound[4], freq > freq_bound[3])
    freq_band5_bool = np.logical_and(freq <= freq_bound[5], freq > freq_bound[4])

    num_preprocess_max = np.size(data_preprocess, 1)
    out_preprocess = np.zeros((14, num_preprocess_max))
    for num_preprocess in range(num_preprocess_max):
        pressure_num = data_preprocess[0, num_preprocess]['Pressure']
        absorbance_num = data_preprocess[0, num_preprocess]['Absorbance']
        magnitude_admittance_num = data_preprocess[0, num_preprocess]['YAdmittance']
        phase_admittance_num = data_preprocess[0, num_preprocess]['PhaseAdmittance']

        number_selected = np.argmin(np.abs(pressure_num))   # Ambient pressure
        absorbance_ambient = absorbance_num[number_selected, :]
        magnitude_admittance_ambient = magnitude_admittance_num[number_selected, :]
        phase_admittance_ambient = phase_admittance_num[number_selected, :]

        out_preprocess[0][num_preprocess] = np.mean(absorbance_ambient[freq_band0_bool])
        out_preprocess[1][num_preprocess] = np.mean(absorbance_ambient[freq_band1_bool])
        out_preprocess[2][num_preprocess] = np.mean(absorbance_ambient[freq_band2_bool])
        out_preprocess[3][num_preprocess] = np.mean(absorbance_ambient[freq_band3_bool])
        out_preprocess[4][num_preprocess] = np.mean(absorbance_ambient[freq_band4_bool])
        out_preprocess[5][num_preprocess] = np.mean(absorbance_ambient[freq_band5_bool])
        out_preprocess[6][num_preprocess] = magnitude_admittance_ambient[0]
        out_preprocess[7][num_preprocess] = magnitude_admittance_ambient[1]
        out_preprocess[8][num_preprocess] = magnitude_admittance_ambient[2]
        out_preprocess[9][num_preprocess] = magnitude_admittance_ambient[3]
        out_preprocess[10][num_preprocess] = phase_admittance_ambient[0]
        out_preprocess[11][num_preprocess] = phase_admittance_ambient[1]
        out_preprocess[12][num_preprocess] = phase_admittance_ambient[2]
        out_preprocess[13][num_preprocess] = phase_admittance_ambient[3]
    return out_preprocess


# data_preprocess_2d function (for ConvNet)
# Recommended values: p_min = -275, p_max = 135, p_stride = 10, kind_interpolate = 'linear' or 'quadratic'
def data_preprocess_2d(data_preprocess, p_min, p_max, p_stride, kind_interpolate):
    "Preprocessing the struct data, and transform input to a <num_sample x num_pres x num_freq> array"
    pressure_nlx = np.arange(p_max, p_min-1, -1*p_stride)
    frequency_nlx1 = data_preprocess['Frequency'][0, 0]
    frequency_nlx2 = data_preprocess['FrequencyAdmittance'][0, 0]
    output_preprocess = np.zeros((np.size(data_preprocess), np.size(pressure_nlx), np.size(frequency_nlx1)+2*4))

    for ii in range(np.size(data_preprocess)):
        pressure_ii = np.transpose(data_preprocess['Pressure'][0, ii])
        absorbance_ii = data_preprocess['Absorbance'][0, ii]
        y_admittance_ii = data_preprocess['YAdmittance'][0, ii]
        phase_admittance_ii = data_preprocess['PhaseAdmittance'][0, ii]

        for jj in range(np.size(frequency_nlx1)):
            f_ip = sip.interp1d(pressure_ii.flatten(), absorbance_ii[:, jj], kind = kind_interpolate)
            y_jj = f_ip(pressure_nlx)
            output_preprocess[ii, :, jj] = y_jj

        for jj1 in range(4):
            f_ip = sip.interp1d(pressure_ii.flatten(), y_admittance_ii[:, jj1], kind = kind_interpolate)
            y_jj = f_ip(pressure_nlx)
            output_preprocess[ii, :, np.size(frequency_nlx1)+jj1] = y_jj

        for jj2 in range(4):
            f_ip = sip.interp1d(pressure_ii.flatten(), phase_admittance_ii[:, jj2], kind = kind_interpolate)
            y_jj = f_ip(pressure_nlx)
            output_preprocess[ii, :, np.size(frequency_nlx1)+4+jj2] = y_jj
    
    pressure_nlx = np.array(np.split(pressure_nlx, np.size(pressure_nlx)))
    output_preprocess = output_preprocess[:, np.newaxis, :, :]
    return output_preprocess, pressure_nlx, frequency_nlx1, frequency_nlx2


# data_labels_generate function
def data_labels_generate(data_patient, data_normal, index_delete_sample=[2,3,12], index_selected_feature=[0,3,6,7]):
    # LR method
    data_p_JM18 = data_preprocess_JM18(data_patient)
    data_n_JM18 = data_preprocess_JM18(data_normal)
    data_all_JM18 = np.append(data_p_JM18, data_n_JM18, axis=1)
    data_lr = np.delete(np.transpose(data_all_JM18[index_selected_feature]), index_delete_sample, axis=0)

    # 2d method
    data_2d_p, _, _, _ = data_preprocess_2d(data_patient, -275, 135, 10, 'linear')
    data_2d_n, _, _, _ = data_preprocess_2d(data_normal, -275, 135, 10, 'linear')
    data_2d = np.delete(np.append(data_2d_p, data_2d_n, axis=0), index_delete_sample, axis=0)

    labels = np.delete(np.append(np.ones(np.size(data_patient), dtype='int32'), np.zeros(np.size(data_normal), dtype='int32')), index_delete_sample, axis=0)
    return data_lr, data_2d, labels


# extract_pca_feature function
def extract_pca_feature(data_original, num):
    data_using = data_original.reshape(data_original.shape[0], int(data_original.size/data_original.shape[0]))

    my_pca = PCA(n_components=num)
    feature_using = my_pca.fit_transform(data_using)

    return feature_using, my_pca


# knn_interpolate function
def knn_interpolate(feature_o, num_k=5, alpha=10):
    num_k = num_k+1             # num_k nearest neighbors
    kdt = KDTree(feature_o, leaf_size=30, metric='euclidean')
    ind = kdt.query(feature_o, k=num_k, return_distance=False)
    x_new = []
    for ii in range(len(ind)):
        x = feature_o[ii]
        ind_x = ind[ii]
        for jj in range(num_k-1):
            xn = feature_o[ind_x[jj+1]]
            x_diff = xn-x
            x_new_temp = np.ones(alpha)[:,np.newaxis]*x[np.newaxis,:]+np.random.rand(alpha)[:,np.newaxis]*x_diff[np.newaxis,:]
            x_new.append(x_new_temp)
    x_new = np.concatenate(x_new)

    return x_new


# wgn_interpolate function
def wgn_interpolate(feature_o, alpha=50, beta=0.2):
    kdt = KDTree(feature_o, leaf_size=30, metric='euclidean')
    dist, _ = kdt.query(feature_o, k=2)
    dist = dist[:,1]
    noise_intensity = np.median(dist)
    mean_chi = np.sqrt(2)*gamma((feature_o.shape[1]+1)/2)/gamma((feature_o.shape[1])/2)

    x_new = []
    for ii in range(feature_o.shape[0]):
        x = feature_o[ii]
        for jj in range(alpha):
            wgn = np.random.randn(feature_o.shape[1])
            wgn_n = wgn/mean_chi
            x_new_temp = x+beta*noise_intensity*wgn_n
            x_new.append(x_new_temp[np.newaxis, :])
    x_new = np.concatenate(x_new)

    return x_new


# mixup_data function
def mixup_data(data, label, alpha=50, beta_value=0.2):
    mixed_data = []
    mixed_label = np.zeros((alpha*data.shape[0], 3))
    for ii in range(alpha*data.shape[0]):
        lam = np.random.beta(beta_value, beta_value)
        index = np.random.permutation(data.shape[0])
        index1, index2 = index[0], index[1]

        mix = lam*data[index1,:]+(1-lam)*data[index2,:]
        label_a, label_b = label[index1], label[index2]

        mixed_data.append(mix[np.newaxis, :])
        mixed_label[ii, 0], mixed_label[ii, 1], mixed_label[ii, 2] = label_a, label_b, lam

    return np.concatenate(mixed_data, axis=0), mixed_label


# generate_synthetic_data function
# Obtain data-level synthetic data
def generate_synthetic_data(data_o, label_o, args_generate, num=24, a=50, k=5, b=0.2, b_mixup=0.2):
    feature_using, my_pca = extract_pca_feature(data_o, num) # PCA feature
    feature_patient = feature_using[label_o==1,:]
    feature_normal = feature_using[label_o==0,:]

    if args_generate == 'mixup':
        feature_synthetic, labels_synthetic = mixup_data(feature_using, label_o, alpha=a, beta_value=b_mixup)
    else:
        if args_generate == 'interpolate':
            feature_p_generate = knn_interpolate(feature_patient, num_k=k, alpha=int(a/k))
            feature_n_generate = knn_interpolate(feature_normal, num_k=k, alpha=int(a/k))

        if args_generate == 'noise':
            feature_p_generate = wgn_interpolate(feature_patient, alpha=a, beta=b)
            feature_n_generate = wgn_interpolate(feature_normal, alpha=a, beta=b)

        feature_synthetic = np.concatenate((feature_p_generate, feature_n_generate), axis=0)
        labels_synthetic = np.concatenate((np.ones(len(feature_p_generate), dtype='int32'), np.zeros(len(feature_n_generate), dtype='int32')), axis=0)

    data_synthetic = my_pca.inverse_transform(feature_synthetic)
    tmp = [data_o.shape[1:][ii] for ii in range(len(data_o.shape[1:]))]
    tmp.insert(0, -1)
    data_synthetic = data_synthetic.reshape(tmp)

    return data_synthetic, labels_synthetic


# one_dataset_split_load function (Using k-fold CV)
def one_dataset_split_load(data_o, labels, skf, size_batch):

    dataloaders_o = []
    for ii, (train_index, test_index) in enumerate(skf.split(data_o, labels)):
        # Spilt train set and test set
        data_o_train, data_o_test = data_o[train_index], data_o[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        # Load data into dataset
        dataset_o_train = data.TensorDataset(torch.from_numpy(data_o_train).float(), torch.from_numpy(labels_train).long())
        dataset_o_test = data.TensorDataset(torch.from_numpy(data_o_test).float(), torch.from_numpy(labels_test).long())

        dataloaders_o_temp = {
            'train': data.DataLoader(dataset_o_train, batch_size=size_batch, shuffle=True, num_workers=0),
            'test': data.DataLoader(dataset_o_test, batch_size=size_batch,shuffle=False, num_workers=0)
        }
        dataset_sizes = {
            'train': len(dataset_o_train),
            'test': len(dataset_o_test)
        }

        dataloaders_o.append(dataloaders_o_temp)
    
    return dataloaders_o, dataset_sizes


# one_syndataset_split_load function (Using k-fold CV)
def one_syndataset_split_load(data_o, labels, skf, args, num, a, k, b, b_mixup, size_batch):

    dataloaders_s = []
    for ii, (train_index, test_index) in enumerate(skf.split(data_o, labels)):
        # Spilt train set and test set
        data_o_train, data_o_test = data_o[train_index], data_o[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        # Obtain synthetic data
        data_s_train, labels_s_train = generate_synthetic_data(data_o_train, labels_train, args, num=num, a=a, k=k, b=b, b_mixup=b_mixup)

        # Load data into dataset
        if args == 'mixup':
            dataset_s_train = data.TensorDataset(torch.from_numpy(data_s_train).float(), torch.from_numpy(labels_s_train).float())
        else:
            dataset_s_train = data.TensorDataset(torch.from_numpy(data_s_train).float(), torch.from_numpy(labels_s_train).long())

        dataset_o_test = data.TensorDataset(torch.from_numpy(data_o_test).float(), torch.from_numpy(labels_test).long())

        dataloaders_s_temp = {
            'train': data.DataLoader(dataset_s_train, batch_size=size_batch, shuffle=True, num_workers=0),
            'test': data.DataLoader(dataset_o_test, batch_size=size_batch,shuffle=False, num_workers=0)
        }
        dataset_sizes = {
            'train': len(dataset_s_train),
            'test': len(dataset_o_test)
        }

        dataloaders_s.append(dataloaders_s_temp)
    
    return dataloaders_s, dataset_sizes


# datasets_split_load function (Using 5-fold CV)
def datasets_split_load(data_input, data_input_2, labels, size_batch=12, args=['interpolate'], fold=5, num=24, a=50, k=5, b=0.2, b_mixup=0.2, s=np.random.randint(1e5)):
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=s) # CV
    dataloaders = []
    dataset_sizes = []
    for ii in range(len(data_input)):
        data_o = data_input[ii]
        dataloaders_o, dataset_sizes = one_dataset_split_load(data_o, labels, skf, size_batch)
        dataloaders.append(dataloaders_o)

    dataloaders_syn = []
    dataset_sizes_syn = []
    for jj in range(len(data_input_2)):
        data_o_2 = data_input_2[jj]
        for kk in range(len(args)):
            dataloaders_s, dataset_sizes_syn = one_syndataset_split_load(data_o_2, labels, skf, args[kk], num, a, k, b, b_mixup, size_batch)
            dataloaders_syn.append(dataloaders_s)
    
    return dataloaders, dataset_sizes, dataloaders_syn, dataset_sizes_syn


# train_model function (Using early stopping for accuracy)
# Train and test a NN model
def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, model_name, epoch_stride, flag_mixup, current_fold=0, stop_early=True):

    # Initialize
    writer = GlobalVar.get_var('writers')[current_fold]       # current_fold+1 represents the current fold number in k-fold CV
    since = time.time()
    print_condition = range(epoch_stride-1, num_epochs, epoch_stride)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_all = {'train':np.zeros(num_epochs), 'test':np.zeros(num_epochs)}
    acc_all = {'train':np.zeros(num_epochs), 'test':np.zeros(num_epochs)}

    print('-'*24)
    print(model_name+' training begins:')
    print()

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion[phase](outputs, labels)
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                if (not flag_mixup) or (phase == 'test'):
                    running_corrects += torch.sum(preds == labels.detach()).double()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            # Deep copy the model
            if stop_early:
                if phase == 'test' and epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            else:
                if phase == 'test' and epoch <= 10 and epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'test' and epoch > 10 and epoch_acc < best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())                

            # Record epoch_loss & epoch_acc
            loss_all[phase][epoch] = epoch_loss
            acc_all[phase][epoch] = epoch_acc
            # TensorBoard Visualization
            writer.add_scalar('{}/Loss/{}'.format(model_name, phase), epoch_loss, epoch)
            writer.add_scalar('{}/Accuracy/{}'.format(model_name, phase), epoch_acc, epoch)
            writer.flush()

        # Print statistics when the condition is met
        if epoch in print_condition:
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-'*12)
            for phase_print in ['train', 'test']:
                print('{}, Loss: {:.4f}, Acc: {:.4f}'.format(phase_print, loss_all[phase_print][epoch], acc_all[phase_print][epoch]))
            print('Time: {:.0f}m {:.2f}s'.format((time.time()-since)//60, (time.time()-since)%60))
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.2f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_all, acc_all


# evaluate_model function
def evaluate_model(model, dataloaders, device, num_class=2):
    outputs_all = []
    labels_all = []
    cf_mat = np.zeros((num_class, num_class))
    model.eval()
    with torch.set_grad_enabled(False):
        # Iterate over data
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            cf_mat += confusion_matrix(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), [x for x in range(num_class)])

            outputs_all.append(outputs)
            labels_all.append(labels)
        
        model_outputs = torch.cat(outputs_all, dim=0)
        true_values = torch.cat(labels_all, dim=0)
        prob_patient = F.softmax(model_outputs, dim=1)[:,1]

        # Calculate auc_score, fpr, tpr
        auc_score = roc_auc_score(true_values.cpu().detach().numpy(), prob_patient.cpu().detach().numpy())
        fpr, tpr, _ = roc_curve(true_values.cpu().detach().numpy(), prob_patient.cpu().detach().numpy())
    
    return auc_score, fpr, tpr, cf_mat


# roc_postprocess function
# Postprocess for roc & auc score
def roc_postprocess(index_evaluating, num_points=100):
    mean_fpr = np.linspace(0, 1, num_points)
    k = len(index_evaluating)

    auc_k_fold = np.zeros(k)
    tpr_k_fold = np.zeros((k, num_points))
    for ii in range(k):
        auc_k_fold[ii] = index_evaluating[ii]['auc']
        tpr_k_fold[ii, :] = np.interp(mean_fpr, index_evaluating[ii]['fpr'], index_evaluating[ii]['tpr'])

    mean_tpr = np.mean(tpr_k_fold, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_k_fold)

    index_roc = np.argmax(mean_tpr-mean_fpr)
    
    std_tpr = np.std(tpr_k_fold, axis=0)
    tpr_upper = np.minimum(mean_tpr+std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr-std_tpr, 0)

    index_evaluating_processed = {
        'mean_fpr':mean_fpr, 'mean_tpr':mean_tpr,
        'mean_auc':mean_auc, 'std_auc':std_auc, 'index_roc':index_roc,
        'tpr_upper':tpr_upper, 'tpr_lower':tpr_lower
    }

    return index_evaluating_processed


# cfm_postprocess function
# Postprocess for some indicators
def cfm_postprocess(cfm, weight=[0.5,0.5]):
    precision_p = cfm[1][1]/(cfm[1][1]+cfm[0][1])
    recall_p = cfm[1][1]/(cfm[1][1]+cfm[1][0])
    F1_p = 2*precision_p*recall_p/(precision_p+recall_p)

    precision_n = cfm[0][0]/(cfm[0][0]+cfm[1][0])
    recall_n = cfm[0][0]/(cfm[0][0]+cfm[0][1])
    F1_n = 2*precision_n*recall_n/(precision_n+recall_n)

    macro_precision = weight[0]*precision_p+weight[1]*precision_n
    macro_recall = weight[0]*recall_p+weight[1]*recall_n
    macro_F1 = 2*macro_precision*macro_recall/(macro_precision+macro_recall)

    accuracy =  (cfm[0][0]+cfm[1][1])/np.sum(cfm)

    indicators_output = {
        'precision_p':precision_p, 'recall_p':recall_p, 'F1_p':F1_p,
        'precision_n':precision_n, 'recall_n':recall_n, 'F1_n':F1_n,
        'w-precision':macro_precision, 'w-recall':macro_recall, 'w-F1':macro_F1, 'accuracy':accuracy
    }

    return indicators_output


# get_parameter_number function
# Calculate the parameter number of a NN
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# plot_net_roc function
def plot_net_roc(index_evaluating_processed, model_name, line_width=1.5, plot_flag=1):
    fpr_plot = index_evaluating_processed['mean_fpr']
    tpr_plot = index_evaluating_processed['mean_tpr']
    index_roc = index_evaluating_processed['index_roc']

    plt.plot(fpr_plot, tpr_plot, label=model_name+r' (AUC = {:.4f}$\pm${:.4f})'.format(index_evaluating_processed['mean_auc'], index_evaluating_processed['std_auc']), lw=line_width)

    if plot_flag == 1 or plot_flag == 2:
        plt.plot(fpr_plot[index_roc], tpr_plot[index_roc], 'kx')

    if plot_flag == 2:
        plt.fill_between(fpr_plot, index_evaluating_processed['tpr_lower'], index_evaluating_processed['tpr_upper'], color='lightgrey', label=r'$\pm$ 1 std. dev.')

    return


# plot_setup function
# Common setup for plot
def plot_setup(x_lim=[-0.04, 1.04], y_lim=[-0.04, 1.04], x_label='x_label', y_label='y_label', save_flag=False, fig_title='title'):

    font = {'family':'serif', 'weight':'normal', 'size':11}

    plt.grid(False)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xticks(fontproperties = 'monospace', size = 11)
    plt.yticks(fontproperties = 'monospace', size = 11)
    plt.xlabel(x_label, font, size=12)
    plt.ylabel(y_label, font, size=12)
    plt.legend(loc="best", prop=font)
    if save_flag:
        plt.savefig(fig_title+'.eps', format='eps', dpi=400, bbox_inches='tight')
    plt.show()

    return


# plot_heatmap function
def plot_heatmap(image, x_lim=[226, 8000], y_lim=[-275, 135], image_size=(6, 4), x_label='Frequency [Hz]', y_label='Pressure [daPa]', save_flag=False, fig_title='title'):

    font = {'family':'serif', 'weight':'normal', 'size':11}

    plt.figure(figsize=image_size)
    heatmap = plt.imshow(image, aspect='auto', interpolation='none', origin='lower', extent=(min(x_lim), max(x_lim), min(y_lim), max(y_lim)))
    plt.colorbar(heatmap)
    heatmap.set_clim(vmin=0, vmax=1)    # Set range of colorbar

    plt.grid(False)
    plt.xticks(fontproperties = 'monospace', size = 10)
    plt.yticks(fontproperties = 'monospace', size = 10)
    plt.xlabel(x_label, font)
    plt.ylabel(y_label, font)
    if save_flag:
        plt.savefig(fig_title+'.eps', format='eps', dpi=400, bbox_inches='tight')
    plt.show()

    return


########################
# Grad-CAM Method
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targetted layers.
    3. Gradients from intermediate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "matchlayer" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        
        return target_activations, x


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, device):
        self.device = device
        self.model = model.to(self.device)
        self.feature_module = feature_module
        self.model.eval()
        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, inputs):
        return self.model(inputs)

    def __call__(self, inputs, index=None, nlx=0):
        features, outputs = self.extractor(inputs.to(self.device))

        if index == None:
            index = np.argmax(outputs.cpu().detach().numpy())
        print('Basis for the prediction as "{:.0f}""'.format(index))

        one_hot = np.zeros((1, outputs.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.device) * outputs)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().detach().numpy()

        target = features[-1]
        target = target.cpu().detach().numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        if np.max(cam) > 0:
          cam = np.maximum(cam, 0)
        else:
          cam = -cam
          nlx = 1
          #print(nlx)
        
        dim = inputs.shape[2:]
        cam = cv2.resize(cam, (dim[1],dim[0]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, index, nlx

def grad_cam_preprocess(dataloaders):

    images_cam = {'train':0, 'test':0}
    classes_cam = {'train':0, 'test':0}
    for phase in ['train', 'test']:
        images_phase = []
        classes_phase = []
        for images_input, images_label in iter(dataloaders[phase]):
            for ii in range(len(images_label)):
                images_phase.append(images_input[ii].unsqueeze(0).requires_grad_(True))
                classes_phase.append(images_label[ii])
        
        images_cam[phase] = images_phase
        classes_cam[phase] = classes_phase
    
    return images_cam, classes_cam


########################
# Others
# Reset pretrained model
def reset_pretrained_model(nets_nn, flag_freeze, fold=5):
    nets_out = copy.deepcopy(nets_nn)
    for jj in range(fold):
        # Decide whether to freeze all the network except the final layer
        if flag_freeze:
            for param in nets_out[jj].parameters():
                param.requires_grad = False
        # Reset the final layer
        nets_out[jj].classifier[2] = nn.Linear(nets_out[jj].classifier[2].in_features, 2)

    return nets_out


# Adapt to MonteCarlo settings
# Postprocess for roc & auc score for MC evaluation
def roc_postprocess_MC(index_evaluating, fold=5, num_points=100):
    mean_fpr = np.linspace(0, 1, num_points)
    k = len(index_evaluating)

    auc_k_fold = np.zeros(k)
    tpr_k_fold = np.zeros((k, num_points))
    for ii in range(k):
        auc_k_fold[ii] = index_evaluating[ii]['auc']
        tpr_k_fold[ii, :] = np.interp(mean_fpr, index_evaluating[ii]['fpr'], index_evaluating[ii]['tpr'])

    mean_tpr = np.mean(tpr_k_fold, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    
    auc_five_fold = np.zeros(int(k/fold))
    tpr_five_fold = np.zeros((int(k/fold), num_points))
    for jj in range(int(k/fold)):
        auc_five_fold[jj] = np.mean(auc_k_fold[fold*jj:fold*(jj+1)])
        tpr_five_fold[jj, :] = np.mean(tpr_k_fold[fold*jj:fold*(jj+1), :], axis=0)

    std_auc = np.std(auc_five_fold)

    index_roc = np.argmax(mean_tpr-mean_fpr)
    
    std_tpr = np.std(tpr_five_fold, axis=0)
    tpr_upper = np.minimum(mean_tpr+std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr-std_tpr, 0)

    index_evaluating_processed = {
        'mean_fpr':mean_fpr, 'mean_tpr':mean_tpr,
        'mean_auc':mean_auc, 'std_auc':std_auc, 'index_roc':index_roc,
        'tpr_upper':tpr_upper, 'tpr_lower':tpr_lower
    }

    return index_evaluating_processed


# Adapt to MonteCarlo settings
# Postprocess for cfm for MC evaluation
def cfm_postprocess_MC(cfm0, fold=5, w=[0.5,0.5]):
    k = len(cfm0)
    temp_indicators = []
    for jj in range(int(k/fold)):
        temp_indicators.append(cfm_postprocess(sum(cfm0[fold*jj:fold*(jj+1)]),weight=w))

    indicators_std = {}
    for key, _ in temp_indicators[0].items():
        temp_value = np.zeros(int(k/fold))
        for jj in range(int(k/fold)):
            temp_value[jj] = temp_indicators[jj][key]
        indicators_std.update({key+'_std':np.std(temp_value)})

    indicators_mean = cfm_postprocess(sum(cfm0),weight=w)

    return indicators_mean, indicators_std




def plot_3d_wbt(img_wbt, x_label='Freq [Hz]', y_label='Pres [daPa]', z_label='Value', off_set = -0.2, save_flag=False, fig_title='title'):

    fig = plt.figure()
    ax3 = plt.axes(projection='3d')

    xx = np.array([[ 226.  ,  257.33,  280.62,  297.3 ,  324.21,  343.49,  363.91,
            385.55,  408.48,  432.77,  458.5 ,  471.94,  500.  ,  514.65,
            545.25,  561.23,  577.68,  594.6 ,  629.96,  648.42,  667.42,
            686.98,  707.11,  727.83,  749.15,  771.11,  793.7 ,  816.96,
            840.9 ,  865.54,  890.9 ,  917.  ,  943.87,  971.53, 1000.  ,
            1029.3 , 1059.46, 1090.51, 1122.46, 1155.35, 1189.21, 1224.05,
            1259.92, 1296.84, 1334.84, 1373.95, 1414.21, 1455.65, 1498.31,
            1542.21, 1587.4 , 1633.92, 1681.79, 1731.07, 1781.8 , 1834.01,
            1887.75, 1943.06, 2000.  , 2058.6 , 2118.93, 2181.02, 2244.92,
            2310.71, 2378.41, 2448.11, 2519.84, 2593.68, 2669.68, 2747.91,
            2828.43, 2911.31, 2996.61, 3084.42, 3174.8 , 3267.83, 3363.59,
            3462.15, 3563.59, 3668.02, 3775.5 , 3886.13, 4000.  , 4117.21,
            4237.85, 4362.03, 4489.85, 4621.41, 4756.83, 4896.21, 5039.68,
            5187.36, 5339.36, 5495.81, 5656.85, 5822.61, 5993.23, 6168.84,
            6349.6 , 6535.66, 6727.17, 6924.29, 7127.19, 7336.03, 7550.99,
            7772.26, 8000.  ]])
    yy = np.arange(-275,145,10)
    X, Y = np.meshgrid(xx, yy)
    Z = img_wbt

    ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap=cm.coolwarm)
    ax3.contourf(X,Y,Z, zdir='z', levels = 8, offset=off_set, cmap='rainbow')

    plt.grid(False)
    ax3.set_xlabel(x_label, labelpad=0)
    ax3.set_ylabel(y_label, labelpad=0)
    ax3.set_zlabel(z_label, labelpad=0)
    ax3.set_zlim(off_set, 1)
    ax3.set_xlim(226, 8000)
    ax3.set_ylim(min(yy), max(yy))
    
    ax3.tick_params(axis='both', which='major', pad=-1)
    ax3.set_xticks([500, 2000, 4000, 6000, 8000])
    ax3.set_yticks([-200, -100, 0, 100])
    ax3.set_zticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    if save_flag:
        plt.savefig(fig_title+'.eps', format='eps', dpi=100, bbox_inches='tight')
    plt.show()

    return


def plot_add_heatmap(img_wbt, img_cam, image_size=(6, 4), lv_contour=np.linspace(0,1,201), lv_2=[0.4,0.6,0.8], str_cmap='rainbow', x_label='Frequency [Hz]', y_label='Pressure [daPa]', colorbar_flag=True, save_flag=False, fig_title='title'):

    font = {'family':'serif', 'weight':'normal', 'size':11}

    xx = np.array([[ 226.  ,  257.33,  280.62,  297.3 ,  324.21,  343.49,  363.91,
            385.55,  408.48,  432.77,  458.5 ,  471.94,  500.  ,  514.65,
            545.25,  561.23,  577.68,  594.6 ,  629.96,  648.42,  667.42,
            686.98,  707.11,  727.83,  749.15,  771.11,  793.7 ,  816.96,
            840.9 ,  865.54,  890.9 ,  917.  ,  943.87,  971.53, 1000.  ,
            1029.3 , 1059.46, 1090.51, 1122.46, 1155.35, 1189.21, 1224.05,
            1259.92, 1296.84, 1334.84, 1373.95, 1414.21, 1455.65, 1498.31,
            1542.21, 1587.4 , 1633.92, 1681.79, 1731.07, 1781.8 , 1834.01,
            1887.75, 1943.06, 2000.  , 2058.6 , 2118.93, 2181.02, 2244.92,
            2310.71, 2378.41, 2448.11, 2519.84, 2593.68, 2669.68, 2747.91,
            2828.43, 2911.31, 2996.61, 3084.42, 3174.8 , 3267.83, 3363.59,
            3462.15, 3563.59, 3668.02, 3775.5 , 3886.13, 4000.  , 4117.21,
            4237.85, 4362.03, 4489.85, 4621.41, 4756.83, 4896.21, 5039.68,
            5187.36, 5339.36, 5495.81, 5656.85, 5822.61, 5993.23, 6168.84,
            6349.6 , 6535.66, 6727.17, 6924.29, 7127.19, 7336.03, 7550.99,
            7772.26, 8000.  ]])
    yy = np.arange(-275,145,10)
    X, Y = np.meshgrid(xx, yy)
    Z1 = img_wbt
    Z2 = img_cam

    plt.figure(figsize=image_size)
    heatmap = plt.contourf(X, Y, Z1, levels=lv_contour, cmap=str_cmap)
    if colorbar_flag:
        plt.colorbar(heatmap)
    
    contour=plt.contour(X, Y, Z2, levels=lv_2, colors='k')
    plt.clabel(contour,fontsize=8,fmt='%.1f')

    plt.grid(False)
    plt.xticks(fontproperties = 'monospace', size = 10)
    plt.yticks(fontproperties = 'monospace', size = 10)
    plt.xlabel(x_label, font)
    plt.ylabel(y_label, font)
    if save_flag:
        plt.savefig(fig_title+'.png', format='png', dpi=100, bbox_inches='tight')
    plt.show()

    return


