import numpy as np
import tensorflow as tf
import math
import pandas as pd
from sklearn import model_selection
import glob
import os
from zipfile import ZipFile
import shutil
import tqdm as tqdm
import random
import argparse

import logging
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

from utils.utils import *
from dataset.dataset import *
from model.model import DistributedModel


parser = argparse.ArgumentParser()
parser.add_argument('-gpus', '--gpus', dest="gpus", default="0")
parser.add_argument('-checkpoint', '--checkpoint', dest="checkpoint", default='../data/checkpoint/checkpoint')
parser.add_argument('-train_dir', '--train_dir', dest="train_dir", default='../data/train/')
parser.add_argument('-train_csv', '--train_csv', dest="train_csv", default='../data/train_labels_0.csv')
parser.add_argument('-test_csv', '--test_csv', dest="test_csv", default='../data/test_labels_0.csv')
options = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= options.gpus

gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(num_gpus, "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

#     policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
#     tf.keras.mixed_precision.experimental.set_policy(policy)
#     print('Compute dtype: %s' % policy.compute_dtype)
#     print('Variable dtype: %s' % policy.variable_dtype)

    
if num_gpus == 0:
    strategy = tf.distribute.OneDeviceStrategy(device='CPU')
    print("Setting strategy to OneDeviceStrategy(device='CPU')")
elif num_gpus == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='GPU')
    print("Setting strategy to OneDeviceStrategy(device='GPU')")
else:
    strategy = tf.distribute.MirroredStrategy()
    print("Setting strategy to MirroredStrategy()")
    
def main():
    input_path = options.train_dir
    file_name = options.train_csv
    query = '*/*.jpg'
    
    train_df, mapping = read_train_file(input_path, file_name, query)
    
    file_name = options.test_csv
    submission_df = read_submission_file(input_path, file_name, query, train=True)    

    config = {
        'learning_rate': 1e-3,
        'momentum': 0.9,
        'scale': 30,
        'margin': 0.1,
        'clip_grad': 10.0,
        'n_epochs': 50,
        'batch_size': 256,
        'input_size': (224, 224, 3),
        'n_classes': len(train_df['label'].unique()),
        'dense_units': 512,
        'dropout_rate': 0.0,
        'save_interval': 50,
        'wandb':False
        }    
    
    test_ds = create_dataset(
            df=submission_df,
            training=False,
            batch_size=config['batch_size'],
            input_size=config['input_size'],
        )    
    
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(config['learning_rate'], momentum=config['momentum'])

        dist_model = DistributedModel(
            input_size=config['input_size'],
            n_classes=config['n_classes'],
            batch_size=config['batch_size'],
            finetuned_weights=options.checkpoint,
            dense_units=config['dense_units'],
            dropout_rate=config['dropout_rate'],
            scale=config['scale'],
            margin=config['margin'],
            optimizer=optimizer,
            strategy=strategy,
            mixed_precision=False,
            clip_grad=config['clip_grad'],
            wandb_log=config['wandb'])

        preds, confs = dist_model.predict(test_ds=test_ds)    
        

    for i, (pred, conf) in enumerate(zip(preds, confs)):
        submission_df.at[i, 'landmark_id'] = f'{mapping[pred]}'
        submission_df.at[i, 'conf'] = f'{conf}'

    submission_df = submission_df.drop('label', axis=1)
    submission_df = submission_df.drop('prob', axis=1)
    submission_df = submission_df.drop('path', axis=1)
    submission_df = submission_df[submission_df.columns[[0, 1, 4]]]
    submission_df.to_csv('../output/submission.csv', index=False)          
    
    
if __name__ == '__main__':
    main()    