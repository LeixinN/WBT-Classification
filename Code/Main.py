#####################################################
#####################################################
# Import packages needed by main scripts

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import torch
import time
import GlobalVar

from torch.utils.tensorboard import SummaryWriter
from file_functions import data_labels_generate, datasets_split_load, reset_pretrained_model, grad_cam_preprocess, plot_net_roc, plot_setup, cfm_postprocess, GradCam, plot_3d_wbt, plot_add_heatmap
from file_nets import ConvNet2
from file_functions2 import train_evaluate_model_CV
from torchsummary import summary
from ptflops import get_model_complexity_info


# Perform Grad-CAM method
def grad_cam_method(model, dataloaders, grad_cam_phase, grad_cam_jj, device):
    "grad_cam_jj -- The number of case (train: 0--107 and test: 0--26)"

    grad_cam_function = GradCam(model=model, feature_module=model.features, target_layer_names=["4"], device=device)
    grad_cam_images, grad_cam_classes = grad_cam_preprocess(dataloaders)
    print('Actual category: "{:.0f}"'.format(grad_cam_classes[grad_cam_phase][grad_cam_jj]))
    input_cases = grad_cam_images[grad_cam_phase][grad_cam_jj]
    mask, index_mask, _ = grad_cam_function(input_cases, index=None)

    if grad_cam_classes[grad_cam_phase][grad_cam_jj]==1 and index_mask==1:
      ss = 'TP'
    if grad_cam_classes[grad_cam_phase][grad_cam_jj]==1 and index_mask==0:
      ss = 'FN'
    if grad_cam_classes[grad_cam_phase][grad_cam_jj]==0 and index_mask==1:
      ss = 'FP'
    if grad_cam_classes[grad_cam_phase][grad_cam_jj]==0 and index_mask==0:
      ss = 'TN'

    return mask, input_cases.cpu().detach().numpy()[0, 0, :], ss

def FT_or_FE(my_tl):
    if my_tl == 'FT':
        tag = False
    if my_tl == 'FE':
        tag = True
    return tag

GlobalVar._init()

GlobalVar.set_var('writers', [SummaryWriter('tb-log/temp/fold-{}'.format(i+1)) for i in range(5)])



#####################################################
#####################################################
# Execute the algorithms programs

# Check if the GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

temp_since = time.time()
# Import data
data_all = sio.loadmat("WBTData.mat")   # Patient data 68, Normal data 55
data_new_added = sio.loadmat("WBTData2.mat")   # Patient data 16

# Preprocess
_, data_2d, labels = data_labels_generate(np.concatenate((data_all['PatientData'], data_new_added['PatientData2']), axis=1), data_all['NormalData'], index_delete_sample=[2,3,12,74], index_selected_feature=[0,3,6,7])
data_2d = data_2d[:, :, :, 0:107]       # Extract absorbance information
print('Total pre-processing time elapsed is {:.0f}m {:.2f}s'.format((time.time()-temp_since)//60, (time.time()-temp_since)%60))

# Set the super-parameters
fold = 5
num_epoch = 40

name_net = ['CNN']
name_syn_data = ['None']    # 'None', 'SynDataI', 'SynDataN', 'SynDataM'
name_tl = ['FT']    # 'FT', 'FE'

param_dataset = {'size_batch':12, 'num_fold':fold, 'num_pca':24, 'num_syn':50, 'num_knn':5, 'noise':0.2, 'mixup':0.2}
param_train = {'fold':fold, 'num_epoch':num_epoch, 'epoch_stride':50}

param = {'name_net':name_net, 'name_syn_data':name_syn_data, 'name_tl':name_tl, 'param_dataset':param_dataset, 'param_train':param_train}


# evaluation
since = time.time()

random_seed = np.random.randint(1e6)

# Prepare nets & dataloaders
nets = {'CNN':[ConvNet2() for i in range(fold)]}
dls_real = {'CNN':[]}
dls_syn = {'SynDataI':[], 'SynDataN':[], 'SynDataM':[]}
# Split & Load
[dls_real['CNN']], dataset_sizes, [dls_syn['SynDataI'], dls_syn['SynDataN'], dls_syn['SynDataM']], dataset_sizes_syn = datasets_split_load([data_2d], [data_2d], labels, size_batch=param_dataset['size_batch'], args=['interpolate', 'noise', 'mixup'], fold=param_dataset['num_fold'], num=param_dataset['num_pca'], a=param_dataset['num_syn'], k=param_dataset['num_knn'], b=param_dataset['noise'], b_mixup=param_dataset['mixup'], s=random_seed)

print('-'*24)
print(summary(nets['CNN'][0], input_size=(1, 42, 107)))
macs, params = get_model_complexity_info(nets['CNN'][0], (1, 42, 107), as_strings=False, print_per_layer_stat=False, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print('-'*24)

# Train & evaluate
eval_cfm = {}
eval_output = {}
for my_net in name_net:
    for my_syn in name_syn_data:
        if my_syn == 'None':
            name = my_net
            rr, cfm_raw, nets_nn, _, outputs_raw = train_evaluate_model_CV(nets[my_net], dls_real[my_net], dataset_sizes, device, name, fold=param_train['fold'], num_epochs=param_train['num_epoch'], epoch_stride=param_train['epoch_stride'])
            eval_cfm.update({name:cfm_raw})
            eval_output.update({name:outputs_raw})
        else:
            for my_tl in name_tl:
                name = my_net+'+'+my_syn+'+'+my_tl

                _, _, nets_pretrain, _, _ = train_evaluate_model_CV(nets[my_net], dls_syn[my_syn], dataset_sizes_syn, device, my_net+'+'+my_syn, fold=param_train['fold'], num_epochs=param_train['num_epoch'], epoch_stride=param_train['epoch_stride'])
                rr, cfm_raw, nets_nn, _, outputs_raw = train_evaluate_model_CV(reset_pretrained_model(nets_pretrain, FT_or_FE(my_tl), fold=param_train['fold']), dls_real[my_net], dataset_sizes, device, name, fold=param_train['fold'], num_epochs=param_train['num_epoch'], epoch_stride=param_train['epoch_stride'])
                eval_cfm.update({name:cfm_raw})
                eval_output.update({name:outputs_raw})


results = {'time':time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'random_seed':random_seed, 'eval_cfm':eval_cfm, 'eval_output':eval_output, 'GPU':device}

print('Total time elapsed is {:.0f}m {:.2f}s'.format((time.time()-since)//60, (time.time()-since)%60))

# Visualize
# Figure a -- Deep Learning
plt.figure()
plot_net_roc(rr, name)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5)
plot_setup(x_label='False Positive Rate', y_label='True Positive Rate', save_flag=False, fig_title='fig_a')

# Display indicators
print()
print('-'*24)
print(name)
print(cfm_postprocess(sum(cfm_raw)))
print('-'*24)

# WBT image and the diagnosis output

dataloaders_gcam = dls_real['CNN']

grad_cam_ii = 0     # The number of fold (0--4)
ii = 14             # (train: 0--107 and test: 0--26)

img_cam, img_wbt, ss = grad_cam_method(nets_nn[grad_cam_ii], dataloaders_gcam[grad_cam_ii], 'test', ii, device)

print(ss)

img_wbt[img_wbt==0]=1e-5

plot_3d_wbt(img_wbt, save_flag=False, fig_title='n_wbt')

plot_add_heatmap(img_wbt, img_cam, save_flag=False, fig_title='img_cam_example')


