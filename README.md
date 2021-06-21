# training-with-s3

This repository contains sample code I used to train pytorch models on [BigEarthNet](http://bigearth.net/). BigEarthNet is a dataset so large that it cannot be loaded into memory when training. One solution is to save training examples as s3 objects that are then streamed into the model at training time. I put this code in a repository since it seems useful and I'm sure I'll need something like it again.

## Preprocess

The script `preprocess.py` processes files into training examples and saves those examples as s3 objects.  Note that this script uses raterio to process the `.tif` files into arrays. Some of the bands have lower resolution. The script interpolates the low-res bands using cubic resampling (this is not necessarily the most efficient thing to do since this increases the file size for each example) and then stacks them into 12 x 120 x 120 tensor. Each tensor is combined with the corresponding label, pickled, and put in an s3 object.

(There is a similar preprocessing script available from [BigEarthNet gitlab](https://git.tu-berlin.de/rsim/BigEarthNet-S2_43-classes_models), but their code does not use s3.)

## Dataset

The file `Dataset.py` subclasses pytorch's Dataset. The `__getitem__` method fetches an object from s3, unpacks the data and the label into torch tensors, and returns them to the training script. 

### Requirements

The first step before preprocessing is to download and unzip the BigEarthNet-S2 file, available at the link above. 

One also needs to have downloaded the train/test/val splits from [BigEarthNet gitlab](https://git.tu-berlin.de/rsim/BigEarthNet-S2_43-classes_models). These .csv files contain the folder names used by the preprocessing script. 

One needs to have set up an s3 bucket and configured their credentials for boto3.