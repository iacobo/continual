#%%

import torch
from sklearn.datasets import make_gaussian_quantiles
# Construct dataset

def generate_experiences(n_timesteps, n_channels, n_tasks, n_samples=1000, test=False):

    if test:
        n_samples //= 5
    experiences = [make_gaussian_quantiles(mean=[2*i for _ in range(n_channels*n_timesteps)],
                                    cov=[3. for _ in range(n_channels*n_timesteps)],
                                    n_samples=n_samples, n_features=n_channels*n_timesteps,
                                    n_classes=2, random_state=1)
                                    for i in range(n_tasks)]

    for i in range(n_tasks):
        x = torch.FloatTensor(experiences[i][0]).view(-1, n_timesteps, n_channels)
        y = torch.LongTensor(experiences[i][1])
        experiences[i] = (x,y)

    return experiences

#############################################################
# Detecting eye open/closed (0/1) based on phys sensors
# Split time-series into chunks, target is mode target of chunk.
#############################################################

from scipy.io import arff
import pandas as pd
import numpy as np
import torch

def grab_eeg_data():
    data = arff.loadarff(r'C:\Users\jacob\Downloads\EEG Eye State.arff')
    df = pd.DataFrame(data[0])
    df['eyeDetection'] = [int(s.decode()) for s in df['eyeDetection']]

    #14980 obs
    partitions = len(df)//8
    trunc = len(df) - len(df)%partitions
    dfs = np.split(df[:trunc], partitions)

    # Creating tensors
    target = [df['eyeDetection'].values for df in dfs]
    target = torch.mode(torch.tensor(target))[0]
    features = torch.FloatTensor([df.drop('eyeDetection', axis=1).values for df in dfs])

    return features, target

def generate_eeg_experiences(n_tasks):
    features, target = grab_eeg_data()
    datalen = features.shape[0]

    if datalen%n_tasks != 0:
        trunc = datalen - datalen%n_tasks
        features, target = features[:trunc], target[:trunc]

    train_size = int(0.8*datalen//n_tasks)
    test_size = datalen//n_tasks - train_size
    task_sizes = [size for _ in range(n_tasks) for size in (train_size,test_size)]
    
    features = features.split(task_sizes)
    target = target.split(task_sizes)

    features = list(features)

    for i in range(0,n_tasks*2,2):
        features[i] = features[i] + torch.ones_like(features[i])*i*100
        features[i+1] = features[i+1] + torch.ones_like(features[i+1])*i*100

    return list(zip(features,target))[::2], list(zip(features,target))[1::2]

def generate_permuted_eeg_experiences(n_tasks):
    features, target = grab_eeg_data()

    perm_feats = [features[:,:,torch.randperm(features.shape[2])] for i in range(n_tasks)]

    datalen = len(target)
    train_size = int(0.8*datalen)
    
    return [(f[:train_size],target[:train_size]) for f in perm_feats], [(f[train_size:],target[train_size:]) for f in perm_feats]
 

# Check exp[1][0] == exp[0][0] shows weird all True col


# %%
