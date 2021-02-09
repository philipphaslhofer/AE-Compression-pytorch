# system packages
import sys
import os
import configparser
BIN = './..' #'/Users/philipphaslhofer/Documents/University/TechnicalPhysics/Master/PW_DarkMatter/code/AE-Compression-pytorch/'
sys.path.append(BIN)

# data & algebra packages
import numpy as np
import pandas as pd
import json

# visualization packages
import matplotlib.pyplot as plt

# intelligent packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

#import fastai
from fastai import learner
from fastai.data import core
from fastai.metrics import mse
from fastai.callback import schedule

### Define Neural Networks
from HEPAutoencoders.nn_utils import AE_basic, AE_bn, AE_LeakyReLU, AE_bn_LeakyReLU, AE_big, AE_3D_50, AE_3D_50_bn_drop, AE_3D_50cone, AE_3D_100, AE_3D_100_bn_drop, AE_3D_100cone_bn_drop, AE_3D_200, AE_3D_200_bn_drop, AE_3D_500cone_bn, AE_3D_500cone_bn, AE_3D_200_LeakyReLU

###------------------------------------------------------------------------------
###------------------------------------------------------------------------------

def configuration():
    global bs
    global bn_wd
    global true_wd
    global wd
    global n_epoch
    global threshold
    global nodes

    config = configparser.ConfigParser()

    if len(sys.argv) < 4:
        print('Reading default configuration file!\n')

        config.read('config.ini')
        default = config['default']
        bs = default.getint('bs')
        bn_wd = default.getboolean('bn_wd')  # Don't use weight decay for batchnorm layers
        true_wd = default.getboolean('true_wd')  # weight decay will be used for all optimizers
        wd = default.getfloat('wd') #0.01 0.001 0.0001
        n_epoch = default.getint('n_epoch')
        nodes = json.loads(config['default']['nodes']) # Setup a NeuralNetwork instance
    else:
        print('Reading custom configuration file!\n')
        config.read(sys.argv[3])
        if len(config.sections()) == 1:
            default = config['default']
            bs = default.getint('bs')
            bn_wd = default.getboolean('bn_wd')  # Don't use weight decay for batchnorm layers
            true_wd = default.getboolean('true_wd')  # weight decay will be used for all optimizers
            wd = default.getfloat('wd') #0.01 0.001 0.0001
            n_epoch = default.getint('n_epoch')
            nodes = json.loads(config['default']['nodes']) # Setup a NeuralNetwork instance
        else:
            custom = config.sections()[1]
            bs = custom.getint('bs')
            bn_wd = custom.getboolean('bn_wd')  # Don't use weight decay for batchnorm layers
            true_wd = custom.getboolean('true_wd')  # weight decay will be used for all optimizers
            wd = custom.getfloat('wd') #0.01 0.001 0.0001
            n_epoch = custom.getint('n_epoch')
            nodes = json.loads(config['default']['nodes']) # Setup a NeuralNetwork instance
        print(custom)

    threshold= 1e-5
    #nodes = [4, 200, 200, 3, 200, 200, 4]

def fetch_data():
    global train, test, data_path
    data_path = sys.argv[2]
    print(f'Fetching data @: {data_path}')

    ## Paths to point to where you have stored the datasets.
    train_path = data_path + '_4D_train.pkl'
    ##/Users/philipphaslhofer/Documents/University/TechnicalPhysics/Master/DarkMatter_Proj/data/phenoML'
    test_path = data_path + '_4D_test.pkl'
    ##/Users/philipphaslhofer/Documents/University/TechnicalPhysics/Master/DarkMatter_Proj/data/phenoML'

    train = pd.read_pickle(train_path)
    print('Training Samples: ', train.size//4)
    test = pd.read_pickle(test_path)

def normalize(energy=1e6, momentum=1e6, eta=5, phi=3):
    ## Custom normalization of variables ##
    test['E'] = test['E'] / energy
    test['pt'] = test['pt'] / momentum

    test['eta'] = test['eta'] / eta
    test['phi'] = test['phi'] / phi


    ##
    train['E'] = train['E'] / energy
    train['pt'] = train['pt'] / momentum

    train['eta'] = train['eta'] / eta
    train['phi'] = train['phi'] / phi

    print('Performed normalization!')

def setup():
    train_x = train
    test_x = test
    train_y = train_x  # y = x since we are building an autoencoder
    test_y = test_x

    # Constructs a tensor object of the data and wraps them in a TensorDataset object.
    train_ds = TensorDataset(torch.tensor(train_x.values, dtype=torch.float), torch.tensor(train_y.values, dtype=torch.float))
    valid_ds = TensorDataset(torch.tensor(test_x.values, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))

    #

    # Converts the TensorDataset into a DataLoader object and combines into one DataLoaders object (a basic wrapper
    # around several DataLoader objects).
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
    global dls
    dls = core.DataLoaders(train_dl, valid_dl)

def setup_networks():
    global network_dict #= dict()
    network_dict = dict()
    network_dict['AE_3D_200_LeakyReLU'] = AE_3D_200_LeakyReLU()
    network_dict['AE_basic'] = AE_basic(nodes)
    network_dict['AE_bn'] = AE_bn(nodes)
    network_dict['AE_LeakyReLU'] = AE_LeakyReLU(nodes)
    network_dict['AE_bn_LeakyReLU'] = AE_bn_LeakyReLU(nodes)
    network_dict['AE_big'] = AE_big()
    network_dict['AE_3D_50'] = AE_3D_50()
    network_dict['AE_3D_50_bn_drop'] = AE_3D_50_bn_drop()
    network_dict['AE_3D_50cone'] = AE_3D_50cone()
    network_dict['AE_3D_100'] = AE_3D_100()
    network_dict['AE_3D_100_bn_drop'] = AE_3D_100_bn_drop()
    network_dict['AE_3D_100cone_bn_drop'] = AE_3D_100cone_bn_drop()
    network_dict['AE_3D_200'] = AE_3D_200()
    network_dict['AE_3D_200_bn_drop'] = AE_3D_200_bn_drop()
    network_dict['AE_3D_500cone_bn'] = AE_3D_500cone_bn()
    network_dict['AE_3D_500cone_bn'] = AE_3D_500cone_bn()

    #return network_dict

def train_model():
    model = network_dict[sys.argv[1]]
    model.train()

    model.to('cpu')
    loss_func = nn.MSELoss()

    model_string = str(model).split('(')[0]
    path_string = data_path.split('/')[-1].split('.')[0]

    print(f'Training on: {path_string} \nUsing model: {model_string}\n')

    learn = learner.Learner(dls, model=model, wd=wd, loss_func=loss_func)
    #wd=None, wd_bn_bias=False, train_bn=True
    print(f'Training!')
    learn.fit_one_cycle(n_epoch=n_epoch)

    model_string = str(model).split('(')[0]
    path_string = data_path.split('/')[-1].split('.')[0]
    save_string = path_string + '-' + model_string + '-' + 'bs' + str(bs) + '_wd' + str(wd) + '_valdata'

    learn.save(save_string)

    save_data(learn, save_string)

    model.eval()

    loss = learn.validate()[0]
    if loss < threshold:
        print(f'Training successfull!\nValidation loss: {loss}')
    else:
        print(f'Training unsuccessfull!\nValidation loss {np.round(loss, 5)} above threshold: {threshold}')

def save_data(learn, save_string):
    save_path = '../../../data/' + save_string + '/'
    save_dir = '../../../data/' + save_string
    print(save_string, save_path, save_dir)
    # Make and save figures
    if not os.path.exists(save_path):
        os.mkdir(save_dir)

    with open(save_path + 'params.txt', 'w') as pf:
        pf.write(   f'bs = {bs}\nbn_wd = {bn_wd}\ntrue_wd = {true_wd}\nwd = {wd}\nn_epoch = {n_epoch}\nnodes = {nodes}'
                )
    # Plot losses
    batches = len(learn.recorder.losses)
    val_iter = (batches / n_epoch) * np.arange(1, n_epoch + 1, 1)
    loss_name = str(learn.loss_func)
    #print(loss_name.split('(')[0])
    plt.figure()
    plt.semilogy(learn.recorder.losses, label='Train')
    plt.legend()
    plt.ylabel(loss_name)
    plt.xlabel('Batches processed')
    plt.savefig(save_path + 'losses.png')
    with open(save_path + 'losses.txt', 'w') as f:
        f.write(f'{save_string};\n')
        for i_val, val in enumerate(learn.recorder.losses):
            f.write(f'{val};\n')
    print(f'\nWrote data to folder: {save_path}')
    return

def main():
    if sys.hexversion < 3.8 :
        raise ValueError('System Python version needs to be 3.8 or newer')
    else:
        print(f'Current version of Python used: {sys.version}')

    ### Change this to accept input provided from a script or command line input
    if len(sys.argv) < 3:
        raise ValueError(f'Two arguments must be passed:/nFirst argument:/tThe network to be trained!/nSecond Argument:/tThe data to be trained on!')

    configuration()

    print(f'Parameters:\nBatchSize: {bs}\nWeightDecay: {wd}\nEpochs: {n_epoch}\n')

    fetch_data()
    normalize()
    setup()
    setup_networks()
    train_model()

###------------------------------------------------------------------------------
###------------------------------------------------------------------------------

if __name__=='__main__':
    main()
