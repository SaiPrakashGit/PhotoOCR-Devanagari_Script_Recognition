from matplotlib.pyplot import imread
import numpy as np
import pandas as pd
import os, sys

TRAIN = './train'
VALIDATE = './valid'

root_directory = VALIDATE

try:
    if sys.argv[1] == 'TRAIN':
        print ("Converting and dumping the training set images into train.csv file...")
        root_directory = TRAIN
    elif sys.argv[1] == 'VALID':
        print ("Converting and dumping the validation set images into valid.csv file...")
    else:
        print ("Invalid argument.. Quitting the program..")
        sys.exit()
except:
    root_directory = VALIDATE
    
# go through each directory in the root folder given above
for directory, subdirectories, files in os.walk(root_directory):
    # go through the each image file in the directory
    for file in files:
        # read the image file and extract its pixel densities
        img = imread(os.path.join(directory, file))
        array = img.flatten()
        # append the correct character/digit label to the array
        array = np.hstack((directory[8:], array))
        df = pd.DataFrame(array).T
        # df = df.sample(frac=1)
        # Write the dataframe into the csv file
        if(root_directory == VALIDATE):
            with open('valid_ordered.csv', 'a') as dataset:
                df.to_csv(dataset, header=False, index=False)
        else:
            with open('train_ordered.csv', 'a') as dataset:
                df.to_csv(dataset, header=False, index=False)
            
# Shuffle the rows in the datasets
if(root_directory == VALIDATE):
    df = pd.read_csv('valid_ordered.csv', header=None)
    dfs = df.sample(frac=1) # Shuffles the dataframe
    dfs.to_csv('valid.csv', header=False, index=False)
else:
    df = pd.read_csv('train_ordered.csv', header=None)
    dfs = df.sample(frac=1) # Shuffles the dataframe
    dfs.to_csv('train.csv', header=False, index=False)