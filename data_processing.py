#%%

import torch
from sklearn.datasets import make_gaussian_quantiles
import seaborn as sns; sns.set()
from scipy.io import arff
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt


# Construct dataset

def generate_experiences(n_timesteps, n_channels, n_tasks, n_samples=2000, test=False):

    if test:
        n_samples //= 5
    experiences = [make_gaussian_quantiles(mean=[3*i for _ in range(n_channels*n_timesteps)],
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

def grab_eeg_data():
    data = arff.loadarff(r'C:\Users\jacob\OneDrive\Documents\code\cl code\ehr\EEG Eye State.arff')
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
    
    # Use sklearn train_test_split
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

#####################################################################

def load_activity_data(test=False):
    if not test:
        df = pd.read_csv(r'C:\Users\jacob\OneDrive\Documents\code\cl code\ehr\data\train.csv')
    else:
        df = pd.read_csv(r'C:\Users\jacob\OneDrive\Documents\code\cl code\ehr\data\test.csv')
    
    df['Target'] = df['Activity'].str.contains('WALKING')
    tasks = [None,None,None]
    tasks[0] = df[df['Activity'].isin(['LAYING','WALKING'])]
    tasks[2] = df[df['Activity'].isin(['SITTING','WALKING_DOWNSTAIRS'])]
    tasks[1] = df[df['Activity'].isin(['STANDING','WALKING_UPSTAIRS'])]

    for i in range(len(tasks)):
        #plt.plot(list(range(len(tasks[i]))), tasks[i]['Activity'])
        #plt.show()

        tasks[i] = tasks[i].drop(['subject', 'Activity'], axis=1)
        tasks[i] = tasks[i][['angle(X,gravityMean)', 'angle(Y,gravityMean)', 'Target']]

        partitions = len(tasks[i])//6
        trunc = len(tasks[i]) - len(tasks[i])%partitions
        dfs = np.split(tasks[i][:trunc], partitions)

        target = [d['Target'].values for d in dfs]
        target = torch.mode(torch.LongTensor(target))[0]
        features = torch.FloatTensor([d.drop('Target', axis=1).values for d in dfs])

        tasks[i] = (features, target)
    return tasks



########################
# eICU DATASET
########################

def load_eicu(drop_dupes=False):
    # Load tables
    df_p = pd.read_csv(r'C:\Users\jacob\OneDrive\Documents\code\cl code\ehr\data\eICU\patient.csv')
    df_h = pd.read_csv(r'C:\Users\jacob\OneDrive\Documents\code\cl code\ehr\data\eICU\hospital.csv')
    # Cleaning
    df_p['age'] = pd.to_numeric(df_p['age'].replace('> 89',90))
    df_p['ethnicity_coarse'] = np.where(df_p['ethnicity']=='Caucasian', 'Caucasian', 'BAME')
    df_p['region'] = pd.merge(df_p,df_h,how='left',on='hospitalid')['region']

    # Grabbing relevant cols
    # 'patientunitstayid', 'patienthealthsystemstayid', 
    df_p = df_p[['uniquepid', 'patientunitstayid', 'patienthealthsystemstayid', 'hospitaldischargeoffset', 'hospitaldischargestatus', 'gender', 'age', 'ethnicity', 'ethnicity_coarse', 'region']]

    # Some patients have multiple admissions in different hospital regions, hence mismatch
    if drop_dupes:
        df_p = df_p[['uniquepid', 'hospitaldischargestatus', 'gender', 'age', 'ethnicity', 'ethnicity_coarse', 'region']]
        len_distinct = len(df_p.drop_duplicates())
        len_ids = len(df_p.groupby('uniquepid'))
        print(f'There are {len_distinct} unique rows, and {len_ids} unique patient ids.')
        df_p = df_p.drop_duplicates()

    return df_p

def plot_demos():

    df = load_eicu(drop_dupes=True)

    fig, axes = plt.subplots(3,2, sharey=True, figsize=(18,18), squeeze=False)

    df['gender'].value_counts().plot.bar(ax=axes[0,0], rot=0, title='Gender')
    df['ethnicity'].value_counts().plot.bar(ax=axes[1,0], rot=0, title='Ethnicity')
    df['ethnicity_coarse'].value_counts().plot.bar(ax=axes[1,1], rot=0, title='Ethnicity (coarse)')
    df['age'].plot.hist(bins=20, label='age', ax=axes[0,1], title='Age')
    df['region'].value_counts().plot.bar(ax=axes[2,0], rot=0, title='Region (North America)')
    df['hospitaldischargestatus'].value_counts().plot.bar(ax=axes[2,1], rot=0, title='Outcome')
    plt.show(); plt.close()

def load_eicu_timeseries():

    df_p = load_eicu()
    df_v = pd.read_csv(r'C:\Users\jacob\OneDrive\Documents\code\cl code\ehr\data\eICU\vitalPeriodic.csv')
    df_v = df_v[['patientunitstayid', 'observationoffset', 'temperature', 'sao2', 'heartrate', 'respiration',
       'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean',
       'pasystolic', 'padiastolic', 'pamean', 'st1', 'st2', 'st3', 'icp',]]
    df = pd.merge(df_p, df_v, on='patientunitstayid')

    #28 day mortality
    df['Target'] = (df['hospitaldischargeoffset'] - df['observationoffset'] < 60) & (df['hospitaldischargestatus'] == 'Expired')

    #df['Target'].value_counts().plot.bar(rot=0, title='Mortality within 24hrs of observation')
    #plt.show(); plt.close()


    return df

def eicu_to_tensor(demographic, balance=False):
    df = load_eicu_timeseries()

    tasks = []

    if demographic == 'gender':
        tasks.append(df[df['gender']=='Male'])
        tasks.append(df[df['gender']=='Female'])
    elif demographic == 'age':
        tasks.append(df[df['age'].between(0,30)])
        tasks.append(df[df['age'].between(30,60)])
        tasks.append(df[df['age'].between(60,100)])
    elif demographic == 'ethnicity':
        tasks.append(df[df['ethnicity']=='Caucasian'])
        tasks.append(df[df['ethnicity']=='African American'])
        tasks.append(df[df['ethnicity']=='Hispanic'])
        tasks.append(df[df['ethnicity']=='Asian'])
        #tasks.append(df[df['ethnicity']=='Native American']) # No mortality
        tasks.append(df[df['ethnicity']=='Other/Unknown'])
    elif demographic == 'ethnicity_coarse':
        tasks.append(df[df['ethnicity_coarse']=='Caucasian'])
        tasks.append(df[df['ethnicity']=='BAME'])
    elif demographic == 'region':
        tasks.append(df[df['region']=='West'])
        tasks.append(df[df['region']=='Midwest'])
        tasks.append(df[df['region']=='South'])
        tasks.append(df[df['region']=='Northeast'])


    for i in range(len(tasks)):
        seq_len = 30

        tasks[i] = tasks[i][['sao2', 'heartrate', 'respiration', 'Target']]

        # Filling missing vals
        tasks[i] = tasks[i].fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

        partitions = len(tasks[i])//seq_len
        trunc = len(tasks[i]) - len(tasks[i])%partitions
        dfs = np.split(tasks[i][:trunc], partitions)

        target = [d['Target'].values for d in dfs]
        target = torch.LongTensor(target).max(axis=1)[0]
        features = torch.FloatTensor([d.drop('Target', axis=1).values for d in dfs])

        if balance:
            count_0, count_1 = target.bincount()

            if count_0 > count_1:
                target, idx = target.topk(count_1*2)
                features = features[idx]
            else:
                target, idx = target.topk(count_0*2, largest=False)
                features = features[idx]
        
        print(f'Task {i} size: {target.shape[0]} (0:{target.bincount()[0]}, 1:{target.bincount()[1]})')


        tasks[i] = (features, target)

    return tasks

# %%

#plot_demos()
#df = load_eicu_timeseries()
#tasks = eicu_to_tensor(demographic='ethnicity', balance=True)

# %%

# %%
