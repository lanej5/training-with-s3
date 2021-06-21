import os
import pickle

import torch
import pandas as pd
import numpy as np

import boto3

from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, patch_names_csv, transform=None):
        
        self.patch_names = pd.read_csv(patch_names_csv, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self, idx):
        # create boto3 client
        s3 = boto3.client('s3')

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get s3 object name using idx
        s3_obj_name = self.patch_names.iloc[idx, 0]

        # get s3 object
        obj = s3.get_object(Bucket='bigearthnet-processed', Key=s3_obj_name)

        # unpickle dictionary
        serialized_obj = obj['Body'].read()
        dict_obj = pickle.loads(serialized_obj)

        image = dict_obj['image']
        label = dict_obj['label']

        X = torch.from_numpy(image).to(torch.float)
        Y = torch.from_numpy(label).to(torch.float)
        
        if self.transform:
            X = self.transform(X)

        return X, Y