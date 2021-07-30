import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
#import seaborn as sns; sns.set()

########################
# eICU DATASET
########################

def load_eicu(drop_dupes=False, root=Path('.')):
    # Load tables
    df_p = pd.read_csv(root / 'patient.csv')
    df_h = pd.read_csv(root / 'hospital.csv')
    
    # Necessary cols
    # 'patienthealthsystemstayid', 
    p_variables = ['uniquepid', 'patientunitstayid', 'hospitalid', 'hospitaldischargeoffset', 'hospitaldischargestatus', 'hospitaldischargeyear', 'gender', 'age', 'ethnicity']
    df_p = df_p[p_variables]
    df_h = df_h[['hospitalid', 'region']]

    # Cleaning
    df_p['age'] = pd.to_numeric(df_p['age'].replace('> 89',90))
    df_p['ethnicity_coarse'] = np.where(df_p['ethnicity']=='Caucasian', 'Caucasian', 'BAME')
    df_p['region'] = pd.merge(df_p,df_h, how='left', on='hospitalid')['region']

    # Some patients have multiple admissions in different hospital regions, hence mismatch
    if drop_dupes:
        df_p = df_p[['uniquepid', 'hospitaldischargestatus', 'gender', 'age', 'ethnicity', 'ethnicity_coarse', 'region']]
        len_distinct = len(df_p.drop_duplicates())
        len_ids = len(df_p.groupby('uniquepid'))
        print(f'There are {len_distinct} unique rows, and {len_ids} unique patient ids.')
        df_p = df_p.drop_duplicates()

    return df_p

def load_eicu_timeseries(target=60, root=Path('.')):

    # JA: Check size of df_v is same as merged size

    df_p = load_eicu(root=root)
    df_v = pd.read_csv(root / 'vitalPeriodic.csv')
    variables = ['patientunitstayid', 'observationoffset', 'sao2', 'heartrate', 'respiration']
    df_v = df_v[variables]
    df = pd.merge(df_p, df_v, on='patientunitstayid')

    #1hr mortality
    df['Target'] = (df['hospitaldischargeoffset'] - df['observationoffset'] < target) & (df['hospitaldischargestatus'] == 'Expired')

    #df['Target'].value_counts().plot.bar(rot=0, title='Mortality within 24hrs of observation')
    #plt.show(); plt.close()

    return df

def grab_tasks(df, demographic):
    tasks = []

    if demographic == 'gender':
        tasks.append(df[df['gender']=='Male'])
        tasks.append(df[df['gender']=='Female'])

    elif demographic == 'age':
        # JA: Find reference for bucketing ages,
        # Or justify given sizes of buckets
        tasks.append(df[df['age'].between(0,30)])
        tasks.append(df[df['age'].between(30,60)])
        tasks.append(df[df['age'].between(60,100)])

    elif demographic == 'ethnicity':
        tasks.append(df[df['ethnicity']=='Caucasian'])
        tasks.append(df[df['ethnicity']=='African American'])
        tasks.append(df[df['ethnicity']=='Hispanic'])
        tasks.append(df[df['ethnicity']=='Asian'])
        #tasks.append(df[df['ethnicity']=='Native American']) # No mortality
        tasks.append(df[df['ethnicity'].isin(['Other/Unknown'])])

    elif demographic == 'ethnicity_coarse':
        tasks.append(df[df['ethnicity_coarse']=='Caucasian'])
        tasks.append(df[df['ethnicity_coarse']=='BAME'])

    elif demographic == 'region':
        tasks.append(df[df['region']=='Northeast'])
        tasks.append(df[df['region']=='West'])
        tasks.append(df[df['region']=='Midwest'])
        tasks.append(df[df['region']=='South'])

    elif demographic == 'hospitaldischargeyear':
        # JA: check years
        tasks.append(df[df['hospitaldischargeyear']==2014])
        tasks.append(df[df['hospitaldischargeyear']==2015])

    return tasks

def eicu_to_tensor(demographic, root=Path('.'), seq_len=30, balance=False):
    df = load_eicu_timeseries(root=root)
    tasks = grab_tasks(df, demographic)
    variables = ['sao2', 'heartrate', 'respiration', 'Target']

    for i in range(len(tasks)):

        tasks[i] = tasks[i][variables]

        # Filling missing vals
        tasks[i] = tasks[i].fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

        # Splitting 2d time-series into sequence 'chunks'
        # JA: Need to split on ID and admission, then pad to quotient length, then split across time dim and concat
        partitions = len(tasks[i])//seq_len
        trunc = len(tasks[i]) - len(tasks[i])%partitions
        
        # Collating chunks as 3d tensor
        dfs = np.split(tasks[i][:trunc], partitions)
        target = [d['Target'].values for d in dfs]
        target = np.array(target)

        target = torch.LongTensor(target).max(axis=1)[0]
        features = torch.FloatTensor([d.drop('Target', axis=1).values for d in dfs])

        # Balance binary classes
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

########################
# MIMIC DATA
########################

# To implement

########################
# iORD DATA
########################

# To implement

########################
# RANDOM DATA
########################

def random_data(seq_len=30, n_vars=3, n_tasks=3, n_samples=30):
    """
    Returns random data of form:

    [
        (
            Features (standard normal): (n_samples,seq_len,n_vars),
            Target (binary):            (n_samples,)
        ),
        ...
    ]
    """
    tasks = [(torch.randn(n_samples,seq_len,n_vars), torch.randint(0,2,(n_samples,))) for _ in range(n_tasks)]
    return tasks

def plot_demos():
    """
    Plots demographic information of eICU dataset.
    """

    df = load_eicu(drop_dupes=True)
    fig, axes = plt.subplots(3,2, sharey=True, figsize=(18,18), squeeze=False)

    df['gender'].value_counts().plot.bar(ax=axes[0,0], rot=0, title='Gender')
    df['ethnicity'].value_counts().plot.bar(ax=axes[1,0], rot=0, title='Ethnicity')
    df['ethnicity_coarse'].value_counts().plot.bar(ax=axes[1,1], rot=0, title='Ethnicity (coarse)')
    df['age'].plot.hist(bins=20, label='age', ax=axes[0,1], title='Age')
    df['region'].value_counts().plot.bar(ax=axes[2,0], rot=0, title='Region (North America)')
    df['hospitaldischargestatus'].value_counts().plot.bar(ax=axes[2,1], rot=0, title='Outcome')
    plt.show()
    plt.close()
