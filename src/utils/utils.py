import glob
import os
import pandas as pd
import numpy as np

def read_submission_file(input_path, file_name, query, alpha=0.5, train=False):
    files_paths = glob.glob(input_path + query)
    mapping = {}
    for path in files_paths:
        mapping[os.path.splitext(path.split('/')[-1])[0]] = path
    df = pd.read_csv(file_name)
    if not train:
        df['path'] = df['id'].map(mapping)
        df['label'] = -1
        df['prob'] = -1
    else:
        df['path'] = df['id'].map(mapping)
        counts_map = dict(
        df.groupby('landmark_id')['path'].agg(lambda x: len(x)))
        df['counts'] = df['landmark_id'].map(counts_map)
        uniques = df['landmark_id'].unique()
        df['label'] = df['landmark_id'].map(dict(zip(uniques, range(len(uniques)))))
        df['prob'] = (
         (1/df.counts**alpha) / (1/df.counts**alpha).max()).astype(np.float32)
    return df

def read_train_file(input_path, file_name, query, alpha=0.5):
    files_paths = glob.glob(input_path + query)
    mapping = {}    
    for path in files_paths:
        mapping[path.split('/')[-1][:-4]] = path
    df = pd.read_csv(file_name)
    df['path'] = df['id'].map(mapping)
    
    counts_map = dict(
        df.groupby('landmark_id')['path'].agg(lambda x: len(x)))
    df['counts'] = df['landmark_id'].map(counts_map)
    df['prob'] = (
         (1/df.counts**alpha) / (1/df.counts**alpha).max()).astype(np.float32)
    uniques = df['landmark_id'].unique()
    df['label'] = df['landmark_id'].map(dict(zip(uniques, range(len(uniques)))))
    return df, dict(zip(range(len(uniques)), uniques))