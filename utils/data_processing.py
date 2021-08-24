import copy
import json
import torch
import sparse
import pandas as pd

from pathlib import Path
from avalanche.benchmarks.generators import tensors_benchmark

DATA_DIR = Path(__file__).parents[1] / 'data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Compute total dataset first, save, load, then do task splits based on column and then drop those cols

########################
# RANDOM DATA
########################

def random_data(seq_len=30, n_vars=3, n_tasks=4, n_samples=100, p_outcome=0.1):
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
    tasks = [(torch.randn(n_samples,seq_len,n_vars), (torch.rand(n_samples) < p_outcome).long()) for _ in range(n_tasks)]
    return tasks

#######################
# ALL
#######################

def load_data(data, demo, validate=False):
    """
    Data of form:
    (
        x:(samples, variables, time_steps), 
        y:(outcome,)
    )
    """

    # JA: Implement "Save tensor as .np object" on first load, load local copy if exists

    if data=='random':
        experiences = random_data()
        test_experiences = copy.deepcopy(experiences)
        weights = None

    elif data in ('fiddle_mimic3', 'fiddle_eicu'):
        tasks = split_tasks_fiddle(data=data.split('_')[-1], demo=demo)

        experiences, test_experiences = split_trainvaltest_fiddle(tasks)
        experiences = [(torch.FloatTensor(feat).to(DEVICE), torch.LongTensor(target).to(DEVICE)) for feat, target in experiences]
        test_experiences = [(torch.FloatTensor(feat).to(DEVICE), torch.LongTensor(target).to(DEVICE)) for feat, target in test_experiences]

        # Class weights for balancing
        class1_count = sum(experiences[0][1]) + sum(experiences[1][1])
        class0_count = len(experiences[0][1]) + len(experiences[1][1]) - class1_count

        weights = class1_count / torch.LongTensor([class0_count, class1_count])

    else:
        print('Unknown data source.')
        raise NotImplementedError

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

    return scenario, n_tasks, n_timesteps, n_channels, weights




##########
# FIDDLE
##########

def get_coarse_ethnicity(df):
    eth_map = {}
    eth_map['ETHNICITY_value:WHITE'] = [c for c in df.columns if c.startswith('ETHNICITY_value:WHITE')]
    eth_map['ETHNICITY_value:ASIAN'] = [c for c in df.columns if c.startswith('ETHNICITY_value:ASIAN')]
    eth_map['ETHNICITY_value:BLACK'] = [c for c in df.columns if c.startswith('ETHNICITY_value:BLACK')]
    eth_map['ETHNICITY_value:HISPANIC'] = [c for c in df.columns if c.startswith('ETHNICITY_value:HISPANIC')]
    eth_map['ETHNICITY_value:OTHER'] = [c for c in df.columns if c.startswith('ETHNICITY_value:') 
                                                              and c not in eth_map['ETHNICITY_value:WHITE'] + 
                                                                           eth_map['ETHNICITY_value:BLACK'] + 
                                                                           eth_map['ETHNICITY_value:ASIAN'] + 
                                                                           eth_map['ETHNICITY_value:HISPANIC']]

    for k,v in eth_map:
        df[k] = df[v].ne(0).any(axis=1) # Mutually exclusive, sum should work

    return df

def recover_admission_time():
    """
    Function to recover datetime info for admission from FIDDLE.
    """
    *_, df_outcome = load_fiddle()
    df_outcome['SUBJECT_ID'] = df_outcome['stay'].str.split('_', expand=True)[0].astype(int)
    df_outcome['stay_number'] = df_outcome['stay'].str.split('_', expand=True)[1].str.replace('episode','').astype(int)

    ## load original MIMIC-III csv
    df_mimic = pd.read_csv(DATA_DIR / 'mimic3' / 'ADMISSIONS.csv', parse_dates=['ADMITTIME'])

    ## grab quarter (season) from data and id
    df_mimic['quarter'] = df_mimic['ADMITTIME'].dt.quarter

    gr = df_mimic.sort_values('ADMITTIME').groupby('SUBJECT_ID')
    df_mimic['stay_number'] = gr.cumcount()+1
    df_mimic = df_mimic[['SUBJECT_ID','stay_number','quarter']]

    return df_outcome.merge(df_mimic, on=['SUBJECT_ID','stay_number'])


    # Groupby.rank -> group by subject_id, rank by admit time, get n for matching

    # One hot encode, then Merge df's

def get_eicu_region(df):
    
    raise NotImplementedError

    

# Save as .json
demo_cols = {
    'mimic3':{
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
    """
    - `data` ['eicu', 'mimic3']
    - `task` ['ARF_4h','ARF_12h','Shock_4h','Shock_12h','mortality_48h']

    features of form N_patients x Seq_len x Features
    """
    data_dir = DATA_DIR / f'FIDDLE_{data}'
    
    with open(data_dir / 'features' / task / 'X.feature_names.json', 'r') as X_file:
        X_feature_names = json.load(X_file)
    with open(data_dir / 'features' / task / 's.feature_names.json', 'r') as s_file:
        s_feature_names = json.load(s_file)

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

def get_modes(x,feat,seq_dim=1):
    """
    For a tensor of shape NxLxF
    Returns modal value for given feature across sequence dim.
    """
    # JA: Check conversion to tnsor, dtype etc
    return torch.LongTensor(x[:,:,feat]).mode(dim=seq_dim)[0].clone().detach().numpy()

def split_tasks_fiddle(data='mimic3', demo='age', task='mortality_48h'):
    """
    Takes FIDDLE format data and given an outcome and demographic,
    splits the input data across that demographic into multiple
    tasks/experiences.
    """
    features_X, features_s, X_feature_names, s_feature_names, df_outcome = load_fiddle(data, task)

    timevar_categorical_demos = []
    static_categorical_demos = []
    static_onehot_demos = ['sex','age','ethnicity','hospital']
    timevar_onehot_demos = []
    
    # if feat is categorical
    if demo in timevar_categorical_demos:
        assert len(demo_cols[data][demo]) == 1, "More than one categorical col specified!"
        demo_cat = X_feature_names.index(demo_cols[data][demo][0])
        tasks = get_modes(features_X,demo_cat)
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
    elif demo=='time_season':
        seasons = recover_admission_time()['quarter']
        tasks_idx = [seasons==i for i in range(1,5)]
    else:
        raise NotImplementedError

    tasks = [(features_X[idx], df_outcome[idx]) for idx in tasks_idx]

    # Displays number of outcomes per train/test/val partition per task
    print([t[1][['partition', 'y_true']].groupby('partition').agg(Total=('y_true','count'), Outcome=('y_true','sum'),) for t in tasks])

    return tasks

def split_trainvaltest_fiddle(tasks, val_as_test=True):
    """
    Takes a dataset of multiple tasks/experiences and splits it into 
    train and val/test sets.

    Assumes FIDDLE style outcome/partition cols in df of outcome values.
    """
    if val_as_test:
        tasks_train = [(t[0][t[1]['partition']=='train'], t[1][t[1]['partition']=='train']['y_true'].values) for t in tasks]
        tasks_test = [(t[0][t[1]['partition']=='val'], t[1][t[1]['partition']=='val']['y_true'].values) for t in tasks]
    else:
        tasks_train = [(t[0][t[1]['partition'].isin(['train','val'])], t[1][t[1]['partition'].isin(['train','val'])]['y_true'].values) for t in tasks]
        tasks_test = [(t[0][t[1]['partition']=='test'], t[1][t[1]['partition']=='test']['y_true'].values) for t in tasks]

    return tasks_train, tasks_test

def get_hospital_ids():
    """
    Gets hospial id's from df cols.
    """
    return list(map(lambda x: x.split(':')[-1].replace('_',''), demo_cols['eicu']['hospital']))

def cache_processed_dataset():
    # Given dataset/demo/outcome
    # Create train and val, and train and test datasets
    # Save as numpy arrays in data/preprocessed/dataset/outcome/demo
    # Load numpy arrays
    return NotImplementedError
