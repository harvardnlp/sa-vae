import sys
import torch
import torch.utils.data as data_utils
import numpy as np
from scipy.io import loadmat
import os
import argparse
import pickle

def load_omniglot(raw_file, output_file, n_validation=2000):
    # set args
    input_size = [1, 28, 28]
    input_type = 'binary'
    dynamic_binarization = True

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
    omni_raw = loadmat(raw_file)

    # train and test data
    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))

    # shuffle train data
    np.random.shuffle(train_data)

    # set train and validation data
    x_train = train_data[:-n_validation]
    x_val = train_data[-n_validation:]
    # binarize val/test (train is dynamically binarized)
    x_val = np.random.binomial(1, x_val)
    x_test = np.random.binomial(1, x_test)
    num_train = len(x_train)
    num_val = len(x_val)
    num_test = len(x_test)
    print('Train/Val/Test')
    print(num_train, num_val, num_test)
    print('saving data...')
    torch.save([torch.from_numpy(x_train).float().contiguous().view(num_train, 1, 28, 28),
                torch.from_numpy(x_val).float().contiguous().view(num_val, 1, 28, 28),
                torch.from_numpy(x_test).float().contiguous().view(num_test, 1, 28, 28)],
               output_file)
    print('done!')


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--raw_file', help="path to chardata.mat file")
    parser.add_argument('--output', help="where to save the hdf5 file")
    args = parser.parse_args(arguments)
    load_omniglot(args.raw_file, args.output)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
