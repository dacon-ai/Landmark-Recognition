import os
import random
import numpy as np
import pandas as pd


from tqdm import tqdm
from glob import glob

from sklearn.model_selection import KFold

configs = {
    'DATA_DIR': '../data/train', 
    'ROOT_DIR': '../data/',
    'fold': [0, 1, 2, 3, 4],
    'seed': 2021,
}
query = '*/*.jpg'

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

seed_everything(configs['seed'])

def read_image_file(input_path, query):
    file_paths = sorted(glob(os.path.join(input_path, query)))
    return file_paths

def get_class_lists(file_paths):
    class_lists = set([file_paths[i].split('/')[-2] for i in range(len(file_paths))])
    return list(class_lists)

def generate_dict(file_paths, classes):
    ids = []
    landmark_ids = []
    path = []
    for i in tqdm(range(len(file_paths))):
        try:
            landmark_ids.append(classes.index(file_paths[i].split('/')[-2]))
            ids.append(file_paths[i].split('/')[-1][:-4])
            path.append(file_paths[i])
        except:
            pass


    gen_dict = {'id': ids,
             'landmark_id': landmark_ids,
             'path': path}    
    return gen_dict

train_file_paths = read_image_file(configs['DATA_DIR'], query)
classes = get_class_lists(train_file_paths)

print(f"Total num of data: {len(train_file_paths)}")
print(f"Total num of classes: {len(classes)}")

train_df = generate_dict(train_file_paths, classes)
train_df = pd.DataFrame(train_df)

kfold = KFold(n_splits=len(configs['fold']), random_state=configs['seed'], shuffle=True)
kfold.get_n_splits(train_df)

for i, (train_index, test_index) in enumerate(kfold.split(train_df)):
    train_df.loc[test_index, 'val_idx'] = i
    
for fold in configs['fold']:
    print(f'fold : {fold}' )
    train_df_fold = train_df.loc[train_df['val_idx'] != fold].reset_index(drop=True)
    test_df = train_df.loc[train_df['val_idx'] == fold].reset_index(drop=True)
    
    train_df_fold.to_csv(configs['ROOT_DIR'] + f'train_labels_{fold}.csv', index=False)
    test_df.to_csv(configs['ROOT_DIR'] + f'test_labels_{fold}.csv', index=False)
    
    print(f'train_{fold}_df: {len(train_df_fold)}')
    print(f'test_{fold}_df: {len(test_df)}')
train_df.to_csv(configs['ROOT_DIR'] + f'train_labels_fold.csv', index=False)    