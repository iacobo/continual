#%%

import copy
import json
import torch
import sparse
import numpy as np
import pandas as pd

from pathlib import Path
from avalanche.benchmarks.generators import tensors_benchmark

DATA_DIR = Path(__file__).parents[1] / 'data'

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Compute total dataset first, save, load, then do task splits based on column and then drop those cols

########################
# eICU DATASET
########################

def load_eicu(drop_dupes=False):
    # Load tables
    df_p = pd.read_csv(DATA_DIR / 'eICU' / 'patient.csv')
    df_h = pd.read_csv(DATA_DIR / 'eICU' / 'hospital.csv')
    
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

# Save as .json
demo_cols = {
    'mimic3':{
        "time_year":[
            "228396_value_Year"
        ],
        "time_month":[
            "228396_value_Year & month"
            #"228396_value_Year, month, & day"
        ],
        "sex":[
            "GENDER_value:F"
            ],
        "age":[
            "AGE_value:(18.032999999999998, 51.561]", 
            "AGE_value:(51.561, 62.419]", 
            "AGE_value:(62.419, 71.504]", 
            "AGE_value:(71.504, 81.24]", 
            "AGE_value:(81.24, 91.4]"
            ],
        "ethnicity":[
            "ETHNICITY_value:ASIAN", 
            "ETHNICITY_value:ASIAN - ASIAN INDIAN", 
            "ETHNICITY_value:ASIAN - CHINESE", 
            "ETHNICITY_value:ASIAN - VIETNAMESE", 
            "ETHNICITY_value:BLACK/AFRICAN", 
            "ETHNICITY_value:BLACK/AFRICAN AMERICAN", 
            "ETHNICITY_value:BLACK/CAPE VERDEAN", 
            "ETHNICITY_value:BLACK/HAITIAN", 
            "ETHNICITY_value:HISPANIC OR LATINO", 
            "ETHNICITY_value:HISPANIC/LATINO - DOMINICAN", 
            "ETHNICITY_value:HISPANIC/LATINO - PUERTO RICAN", 
            "ETHNICITY_value:MIDDLE EASTERN", 
            "ETHNICITY_value:MULTI RACE ETHNICITY", 
            "ETHNICITY_value:OTHER", 
            "ETHNICITY_value:PATIENT DECLINED TO ANSWER", 
            "ETHNICITY_value:PORTUGUESE", 
            "ETHNICITY_value:UNABLE TO OBTAIN", 
            "ETHNICITY_value:UNKNOWN/NOT SPECIFIED", 
            "ETHNICITY_value:WHITE", 
            "ETHNICITY_value:WHITE - BRAZILIAN", 
            "ETHNICITY_value:WHITE - OTHER EUROPEAN", 
            "ETHNICITY_value:WHITE - RUSSIAN"
            ]
        },
    
    'eicu':{
        "sex":[
            "gender_value:Female", 
            "gender_value:Male"
            ], 
        "age":[
            "age_value:(-0.001, 51.0]", 
            "age_value:(51.0, 61.0]", 
            "age_value:(61.0, 69.0]", 
            "age_value:(69.0, 78.0]", 
            "age_value:(78.0, 89.0]", 
            "age_value:> 89"
            ],
        "ethnicity":[
            "ethnicity_value:African American", 
            "ethnicity_value:Asian", 
            "ethnicity_value:Caucasian", 
            "ethnicity_value:Hispanic", 
            "ethnicity_value:Other/Unknown"
            ],
        "hospital":[
            "hospitalid_value:73__",
            "hospitalid_value:110__", 
            "hospitalid_value:122__", 
            "hospitalid_value:142__", 
            "hospitalid_value:167__", 
            "hospitalid_value:176__", 
            "hospitalid_value:183__", 
            "hospitalid_value:188__", 
            "hospitalid_value:197__", 
            "hospitalid_value:199__", 
            "hospitalid_value:208__", 
            "hospitalid_value:243__", 
            "hospitalid_value:252__", 
            "hospitalid_value:264__", 
            "hospitalid_value:281__", 
            "hospitalid_value:283__", 
            "hospitalid_value:300__", 
            "hospitalid_value:338__", 
            "hospitalid_value:345__", 
            "hospitalid_value:394__", 
            "hospitalid_value:400__", 
            "hospitalid_value:411__", 
            "hospitalid_value:416__", 
            "hospitalid_value:417__", 
            "hospitalid_value:420__", 
            "hospitalid_value:443__", 
            "hospitalid_value:449__", 
            "hospitalid_value:458__"
            ],
        "unit":[
            "unittype_value:CCU-CTICU", 
            "unittype_value:CSICU", 
            "unittype_value:CTICU", 
            "unittype_value:Cardiac ICU", 
            "unittype_value:MICU", 
            "unittype_value:Med-Surg ICU", 
            "unittype_value:Neuro ICU", 
            "unittype_value:SICU"
        ],
        "ward":[ 
            "wardid_value:236__", 
            "wardid_value:607__", 
            "wardid_value:646__", 
            "wardid_value:653__"
            ]
        }
    }

def load_fiddle(data='mimic3', task='mortality_48h', n=50000):
    '''
    - `data` ['eicu', 'mimic3']
    - `task` ['ARF_4h','ARF_12h','Shock_4h','Shock_12h','mortality_48h']

    features of form N_patients x Seq_len x Features
    '''
    data_dir = DATA_DIR / f'FIDDLE_{data}'
    
    with open(data_dir / 'features' / task / 'X.feature_names.json', 'r') as X_file:
        X_feature_names = json.load(X_file)
    with open(data_dir / 'features' / task / 's.feature_names.json', 'r') as s_file:
        s_feature_names = json.load(s_file)

    ## TESTING
    TEST_DATE_COLS = False
    if TEST_DATE_COLS:
        feat_subset = [X_feature_names.index(c) for c in X_feature_names if c.startswith('228396')]
        X_feature_names = [c for c in X_feature_names if c.startswith('228396')]
        #df = pd.DataFrame(sparse.load_npz(data_dir / 'features' / task / 'X.npz')[:,:,feat_subset].todense(), columns=X_feature_names)
        vals = sparse.load_npz(data_dir / 'features' / task / 'X.npz')[:,:,feat_subset].todense()
        return vals

    # Subset vars with list to reduce mem overhead
    var_X_demos = [X_feature_names.index(col) for key, cols in demo_cols[data].items() for col in cols if key.startswith('time')]
    var_X_subset = sorted(list(set(range(10)).union(set(var_X_demos))))
    X_feature_names = [X_feature_names[i] for i in var_X_subset]

    var_s_demos = [s_feature_names.index(col) for key, cols in demo_cols[data].items() for col in cols if not key.startswith('time')]
    var_s_subset = sorted(list(set(range(10)).union(set(var_s_demos))))
    s_feature_names = [s_feature_names[i] for i in var_s_subset]

    # Loading tensors
    features_X = sparse.load_npz(data_dir / 'features' / task / 'X.npz')[:n,:,var_X_subset].todense()
    features_s = sparse.load_npz(data_dir / 'features' / task / 's.npz')[:n,var_s_subset].todense()
    
    df_outcome = pd.read_csv(data_dir / 'population' / f'{task}.csv')[:n]

    return features_X, features_s, X_feature_names, s_feature_names, df_outcome
    #raise NotImplementedError

def get_modes(x,feat,seq_dim=1):
    """
    For a tensor of shape NxLxF
    Returns modal value for given feature across sequence dim.
    """
    # JA: Check conversion to tnsor, dtype etc
    return torch.mode(torch.tensor(x[:,:,feat]), dim=seq_dim)[0].cpu().detach().numpy()

def split_tasks_fiddle(data='mimic3', demo='age', task='mortality_48h'):
    features_X, features_s, X_feature_names, s_feature_names, df_outcome = load_fiddle(data, task)

    timevar_categorical_demos = ['time_year','time_month']
    static_categorical_demos = []
    static_onehot_demos = ['sex','age','ethnicity','hospital']
    timevar_onehot_demos = []
    
    # if feat is categorical
    if demo in timevar_categorical_demos:
        assert len(demo_cols[data][demo]) == 1, "More than one categorical col specified!"
        demo_cat = X_feature_names.index(demo_cols[data][demo][0])
        tasks = get_modes(features_X,demo_cat)
        print(pd.DataFrame(tasks).value_counts())
        tasks_idx = [tasks==i for i in range(min(tasks), max(tasks)+1)]
    elif demo in static_categorical_demos:
        demo_cat = s_feature_names.index(demo_cols[data][demo][0])
        tasks = features_s[:,demo_cat]
        tasks_idx = [tasks==i for i in range(min(tasks), max(tasks)+1)]
    elif demo in timevar_onehot_demos:
        demo_onehots = [X_feature_names.index(col) for col in demo_cols[data][demo]]
        tasks_idx = [get_modes(features_X,i)==1 for i in demo_onehots]
    elif demo in static_onehot_demos:
        demo_onehots = [s_feature_names.index(col) for col in demo_cols[data][demo]]
        tasks_idx = [features_s[:,i]==1 for i in demo_onehots]
    else:
        raise NotImplementedError

    #df_outcome['y_true']
    tasks = [(features_X[idx], df_outcome[idx]) for idx in tasks_idx]

    return tasks

def split_trainvaltest_fiddle(tasks, validate=False):
    tasks_train = [(t[0][t[1]['partition']=='train'], t[1][t[1]['partition']=='train']) for t in tasks]
    if validate:
        tasks_val = [(t[0][t[1]['partition']=='val'], t[1][t[1]['partition']=='val']) for t in tasks[:2]]
    tasks_test = [
        (t[0][t[1]['partition']=='test'], t[1][t[1]['partition']=='test']) if i<2 else 
        (t[0][t[1]['partition'].isin(['val','test'])], t[1][t[1]['partition'].isin(['val','test'])]) for i, t in enumerate(tasks)]
    # take random id's in proportion 80:10:10 from first 2 tasks, ensure outcome label balance.
    # take random id's in proportion 80:20 from subsequent tasks, ensure outcome label balance.
    if validate:
        return tasks_train, tasks_val, tasks_test
    else:
        return tasks_train, tasks_test

#%%
tasks = split_tasks_fiddle(demo='age')
[t[1][['partition', 'y_true']].groupby('partition').agg(Total=('y_true','count'), Outcome=('y_true','sum'),) for t in tasks]

train_exp, val_exp, test_exp = split_trainvaltest_fiddle(tasks, validate=True)
# %%

# Hospital id's
# list(map(lambda x: x.split(':')[-1].replace('_',''), demo_cols['eicu']['hospital']))