import tensorflow as tf
import os, sys
father_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(father_dir)
import tensorflow as tf
from dps.utils.data_provider import DataProvider
from dps.utils import utils
import cv2
import numpy as np
from src.config import *

conf_dirs = ['dps.config.conf_tag_cnn_lstm']


if __name__ == '__main__':
    args = utils.load_argparser_args(conf_dirs)
    # step 1: prepare data
    data_shape = (DATA_HEIGHT, DATA_WIDTH, 3, MAX_LSTM_STEP)
    label_shape = (MAX_LSTM_STEP)
    args.num_classes = 7
    dataProvider = DataProvider(data_shape, label_shape, args.num_classes)
    data_file_name = '../data/train_data_M93A.hdf5'
    dataProvider.load_h5py(data_file_name)
    data, label = dataProvider.next_batch(1)
    print(np.shape(data))
    # step 2: import model
    module = __import__(args.model_path, fromlist=[args.model_name])
    model_class = module.__dict__[args.model_name]
    model = model_class(args)
    vt, vs = model.model_setup(args)
    

    


   
