import copy
import torch
import sparse
import numpy as np
import pandas as pd

from pathlib import Path
from avalanche.benchmarks.generators import tensors_benchmark

DATA_DIR = Path(__file__).parents[1] / 'data' / 'eICU'

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Compute total dataset first, save, load, then do task splits based on column and then drop those cols

########################
# eICU DATASET
########################

def load_eicu(drop_dupes=False, root=DATA_DIR):
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

def load_eicu_timeseries(target=60):

    # JA: Check size of df_v is same as merged size

    df_p = load_eicu()
    df_v = pd.read_csv(DATA_DIR / 'vitalPeriodic.csv')
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

def eicu_to_tensor(demographic, seq_len=30, balance=False):
    df = load_eicu_timeseries()
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

#######################
# ALL
#######################

# PUT THIS IN DATA_PROCESSING
# PUT OTHER MODULES IN utils.___
def load_data(data, demo, validate=False):
    """
    Data of form:
    (
        x:(samples, variables, time_steps), 
        y:(outcome,)
    )
    """

    # JA: Implement "Save tensor as .np object" on first load, load local copy if exists
    data_dir = DATA_DIR / data

    if data=='eICU':
        experiences = eicu_to_tensor(demographic=demo, balance=True)
        test_experiences = copy.deepcopy(experiences)

    elif data=='random':
        experiences = random_data()
        test_experiences = copy.deepcopy(experiences)

    elif data=='MIMIC': raise NotImplemented
    elif data=='iORD': raise NotImplemented
    else:
        print('Unknown data source.')
        pass

    if validate:
        # Make method to return train/val for 'validate==True' and train/test else
        experiences = experiences[:2]
        test_experiences = test_experiences[:2]

    n_tasks = len(experiences)
    n_timesteps = experiences[0][0].shape[-2]
    n_channels = experiences[0][0].shape[-1]

    scenario = tensors_benchmark(
        train_tensors=experiences,
        test_tensors=test_experiences,
        task_labels=[0 for _ in range(n_tasks)],  # Task label of each train exp
        complete_test_set_only=False
    )
    # Investigate from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset

    return scenario, n_tasks, n_timesteps, n_channels




##########
# FIDDLE
##########

def load_fiddle(data, task):
    '''
    - `data` ['eicu', 'mimic']
    - `task` ['ARF_4h','ARF_12h','Shock_4h','Shock_12h','mortality_48h']

    features of form N_patients x Seq_len x Features
    '''
    data_dir = DATA_DIR / f'FIDDLE_{data}'
    features_x = sparse.load_npz(data_dir / 'features' / task / 'X.npz')
    features_s = sparse.load_npz(data_dir / 'features' / task / 's.npz')
    outcome = pd.read_csv(data_dir / 'population' / f'{task}.csv')

    #return features_x, features_s, outcome
    raise NotImplementedError

demo_cols = {'demo1':[0,2,5], 'demp2':[1,3,4]}

def split_fiddle(data, task, demo):
    features_x, features_s, outcome = load_fiddle(data, task)
    tasks_idx = [features_s[i]==1 for i in demo_cols[demo]]
    tasks = [(features_x[idx], outcome[idx]) for idx in tasks_idx]

    #return tasks
    raise NotImplementedError

def train_val_test_split(tasks):
    #assert len(tasks) > 2
    # take random id's in proportion 80:10:10 from first 2 tasks,    ensure outcome label balance.
    # take random id's in proportion 80:10:10 from subsequent tasks, ensure outcome label balance.
    raise NotImplementedError
