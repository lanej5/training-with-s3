import os
import json
import rasterio
from rasterio.enums import Resampling
import numpy as np
import pandas as pd
import uuid
import boto3
import pickle

BAND_NAMES = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
NUM_LABELS = 43
LABEL_INDICES = {'Continuous urban fabric': 0,
 'Discontinuous urban fabric': 1,
 'Industrial or commercial units': 2,
 'Road and rail networks and associated land': 3,
 'Port areas': 4,
 'Airports': 5,
 'Mineral extraction sites': 6,
 'Dump sites': 7,
 'Construction sites': 8,
 'Green urban areas': 9,
 'Sport and leisure facilities': 10,
 'Non-irrigated arable land': 11,
 'Permanently irrigated land': 12,
 'Rice fields': 13,
 'Vineyards': 14,
 'Fruit trees and berry plantations': 15,
 'Olive groves': 16,
 'Pastures': 17,
 'Annual crops associated with permanent crops': 18,
 'Complex cultivation patterns': 19,
 'Land principally occupied by agriculture, with significant areas of natural vegetation': 20,
 'Agro-forestry areas': 21,
 'Broad-leaved forest': 22,
 'Coniferous forest': 23,
 'Mixed forest': 24,
 'Natural grassland': 25,
 'Moors and heathland': 26,
 'Sclerophyllous vegetation': 27,
 'Transitional woodland/shrub': 28,
 'Beaches, dunes, sands': 29,
 'Bare rock': 30,
 'Sparsely vegetated areas': 31,
 'Burnt areas': 32,
 'Inland marshes': 33,
 'Peatbogs': 34,
 'Salt marshes': 35,
 'Salines': 36,
 'Intertidal flats': 37,
 'Water courses': 38,
 'Water bodies': 39,
 'Coastal lagoons': 40,
 'Estuaries': 41,
 'Sea and ocean': 42}

def load_image_from_tif(root_folder, patch_name):
    # returns image as a torch tensor dims 12 x 120 x 120
    bands = []
        
    # read bands from geotiff to numpy array using rasterio
    # all bands are scaled to 120 x 120 pixels using cubic resampling
    for band_name in BAND_NAMES:
        band_path = os.path.join(root_folder, patch_name, patch_name + '_' + band_name + '.tif')
        band_ds = rasterio.open(band_path)
        band_data = band_ds.read(1, out_shape=(120,120),resampling=Resampling.cubic)
        bands.append(np.array(band_data).astype(int))
        
    # combine bands into a single tensor with dims 12 x 120 x 120
    image = np.stack(bands,axis=0)
        
    return image

def load_label_from_json(root_folder, patch_name):
    # returns a one-hot vector as a torch tensor
    
    # load label
    label_path = os.path.join(root_folder, patch_name, patch_name + '_labels_metadata.json')
    
    with open(label_path, 'rb') as f:
        label_json = json.load(f)
        
    labels = label_json['labels']
        
    # convert to one-hot vector
    labels_one_hot = np.zeros(NUM_LABELS)
    
    for label in labels:
        labels_one_hot[LABEL_INDICES[label]] = 1
            
    return labels_one_hot

def preprocess_split(root_folder_in, split_name, patch_names_csv):
    # load data, package into dict, put as s3 object
    # save s3 object names to csv

    if not os.path.exists(root_folder_in):
        print("Folder ", root_folder_in, " does not exist.")  
        return

    s3 = boto3.client('s3')

    df = pd.read_csv(patch_names_csv, header=None)
    df.columns = ['patch_name']
    df['s3_object_name'] = ''

    for _, row in df.iterrows():
        
        # load and preprocess image and label
        image = load_image_from_tif(root_folder_in, row['patch_name'])
        label = load_label_from_json(root_folder_in, row['patch_name'])

        # put in dictionary and pickle
        example = {'image':image, 'label':label}
        serialized_example = pickle.dumps(example)
        
        # create s3-object name (with random prefix to improve performance)
        s3_object_name = ''.join([str(uuid.uuid4().hex[:6]), row['patch_name']])
        row['s3_object_name'] = s3_object_name
        
        # put into bucket
        s3.put_object(Bucket = 'bigearthnet-processed', Key=s3_object_name, Body=serialized_example)

    df.to_csv('splits/' + split_name + '_s3.csv')

    return

def main():
    preprocess_split("BigEarthNet-v1.0","train","splits/train.csv")
    preprocess_split("BigEarthNet-v1.0","val","splits/val.csv")
    preprocess_split("BigEarthNet-v1.0","test","splits/test.csv")
    return

if __name__ == "__main__":
    main()