"""
Functions for loading data.

Loads raw data into several "experiences" (tasks)
for continual learning training scenario.

Tasks split by given demographic.

Loads:

- MIMIC-III - ICU time-series data
- eICU-CRD  - ICU time-series data
- random    - sequential data
"""

import copy
import json
import pandas as pd
import numpy as np
from pathlib import Path

import torch
import sparse
import random
from avalanche.benchmarks.generators import tensors_benchmark

DATA_DIR = Path(__file__).parents[2] / "data"

# Reproducibility
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# JA: Save as .json?
DEMO_COL_PREFIXES = {
    "mimic3": {
        "sex": "GENDER_value:F",
        "age": "AGE_value:",
        "ethnicity": "ETHNICITY_value:",
        "ethnicity_coarse": "ETHNICITY_COARSE_value:",
        "ward": "FIRST_CAREUNIT_value:",
    },
    "eicu": {
        "sex": "gender_value:",
        "age": "age_value:",
        "ethnicity": "ethnicity_value:",
        "hospital": "hospitalid_value:",
        "ward": "unittype_value:",
    },
}

########################
# Simulated (random) DATA
########################


def random_data(seq_len=48, n_vars=6, n_tasks=3, n_samples=150, p_outcome=0.1):
    """
    Returns a sequence of random sequential data and associated binary targets.
    """
    tasks = [
        (
            torch.randn(n_samples, seq_len, n_vars),
            (torch.rand(n_samples) < p_outcome).long(),
        )
        for _ in range(n_tasks)
    ]
    return tasks


#######################
# ALL
#######################


def cache_processed_dataset():
    # Given dataset/demo/outcome
    # Create train and val, and train and test datasets
    # Save as numpy arrays in data/preprocessed/dataset/outcome/demo
    # Load numpy arrays
    return NotImplementedError


def load_data(data, demo, outcome, validate=False):
    """
    Data of form:
    (
        x:(samples, variables, time_steps),
        y:(outcome,)
    )
    """

    # JA: Implement "Save tensor as .np object" on first load, load local copy if exists

    if data == "random":
        experiences = random_data()
        test_experiences = copy.deepcopy(experiences)
        weights = None

    elif data in ("mimic3", "eicu"):
        tasks = split_tasks_fiddle(data, demo, outcome)

        experiences, test_experiences = split_trainvaltest_fiddle(
            tasks, print_task_partitions=not validate
        )
        experiences = [
            (torch.FloatTensor(feat), torch.LongTensor(target))
            for feat, target in experiences
        ]
        test_experiences = [
            (torch.FloatTensor(feat), torch.LongTensor(target))
            for feat, target in test_experiences
        ]

        # Class weights for balancing
        class1_count = sum(experiences[0][1]) + sum(experiences[1][1])
        class0_count = len(experiences[0][1]) + len(experiences[1][1]) - class1_count

        weights = class1_count / torch.LongTensor([class0_count, class1_count])

    if validate:
        experiences = experiences[:2]
        test_experiences = test_experiences[:2]

    else:
        # Cap n tasks
        experiences = experiences[:20]
        test_experiences = test_experiences[:20]

    # Do not use validation sets for training
    if not validate and len(experiences) > 5:
        experiences = experiences[2:]
        test_experiences = test_experiences[2:]

    n_tasks = len(experiences)
    n_timesteps = experiences[0][0].shape[-2]
    n_channels = experiences[0][0].shape[-1]

    scenario = tensors_benchmark(
        train_tensors=experiences,
        test_tensors=test_experiences,
        task_labels=[0 for _ in range(n_tasks)],  # Task label of each train exp
        complete_test_set_only=False,
    )
    # JA: Investigate from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset

    return scenario, n_tasks, n_timesteps, n_channels, weights


##########
# FIDDLE
##########


def get_ethnicity_coarse(data, outcome):
    """
    MIMIC-3 has detailed ethnicity values, but some of these groups have no mortality data.
    Hence create broader groups to get better binary class balance of tasks.
    """

    features_X, features_s, X_feature_names, s_feature_names, df_outcome = load_fiddle(
        data=data, outcome=outcome
    )

    eth_map = {}
    eth_map["ETHNICITY_COARSE_value:WHITE"] = [
        c for c in s_feature_names if c.startswith("ETHNICITY_value:WHITE")
    ]
    eth_map["ETHNICITY_COARSE_value:ASIAN"] = [
        c for c in s_feature_names if c.startswith("ETHNICITY_value:ASIAN")
    ]
    eth_map["ETHNICITY_COARSE_value:BLACK"] = [
        c for c in s_feature_names if c.startswith("ETHNICITY_value:BLACK")
    ]
    eth_map["ETHNICITY_COARSE_value:HISPA"] = [
        c for c in s_feature_names if c.startswith("ETHNICITY_value:HISPANIC")
    ]
    eth_map["ETHNICITY_COARSE_value:OTHER"] = [
        c
        for c in s_feature_names
        if c.startswith("ETHNICITY_value:")
        and c
        not in eth_map["ETHNICITY_COARSE_value:WHITE"]
        + eth_map["ETHNICITY_COARSE_value:BLACK"]
        + eth_map["ETHNICITY_COARSE_value:ASIAN"]
        + eth_map["ETHNICITY_COARSE_value:HISPA"]
    ]

    for k, cols in eth_map.items():
        s_feature_names.append(k)
        idx = [s_feature_names.index(col) for col in cols]
        features_s = np.append(
            features_s, features_s[:, idx].any(axis=1)[:, np.newaxis], axis=1
        )

    return features_X, features_s, X_feature_names, s_feature_names, df_outcome


def recover_admission_time(data, outcome):
    """
    Function to recover datetime info for admission from FIDDLE.
    """
    *_, df_outcome = load_fiddle(data, outcome)
    df_outcome["SUBJECT_ID"] = (
        df_outcome["stay"].str.split("_", expand=True)[0].astype(int)
    )
    df_outcome["stay_number"] = (
        df_outcome["stay"]
        .str.split("_", expand=True)[1]
        .str.replace("episode", "")
        .astype(int)
    )

    # load original MIMIC-III csv
    df_mimic = pd.read_csv(
        DATA_DIR / "FIDDLE_mimic3" / "ADMISSIONS.csv", parse_dates=["ADMITTIME"]
    )

    # grab quarter (season) from data and id
    df_mimic["quarter"] = df_mimic["ADMITTIME"].dt.quarter

    admission_group = df_mimic.sort_values("ADMITTIME").groupby("SUBJECT_ID")
    df_mimic["stay_number"] = admission_group.cumcount() + 1
    df_mimic = df_mimic[["SUBJECT_ID", "stay_number", "quarter"]]

    return df_outcome.merge(df_mimic, on=["SUBJECT_ID", "stay_number"])


def get_eicu_region(df):
    raise NotImplementedError


def load_fiddle(data, outcome, n=None, vitals_only=True):
    """
    - `data`: ['eicu', 'mimic3']
    - `task`: ['ARF_4h','ARF_12h','Shock_4h','Shock_12h','mortality_48h']
    - `n`:    number of samples to pick

    features of form N_patients x Seq_len x Features
    """
    data_dir = DATA_DIR / f"FIDDLE_{data}"

    with open(
        data_dir / "features" / outcome / "X.feature_names.json", encoding="utf-8"
    ) as X_file:
        X_feature_names = json.load(X_file)
    with open(
        data_dir / "features" / outcome / "s.feature_names.json", encoding="utf-8"
    ) as s_file:
        s_feature_names = json.load(s_file)

    # Take only subset of vars to  reduce mem overhead
    if data == "eicu":
        vitals = ["Vital Signs|"]
    elif data == "mimic3":
        vitals = ["HR", "RR", "SpO2", "SBP", "Heart Rhythm", "SysBP", "DiaBP"]

    vital_col_ids = [
        X_feature_names.index(var)
        for var in X_feature_names
        for prefix in vitals
        if var.startswith(prefix)
    ]

    if vitals_only:
        X_feature_names = [X_feature_names[i] for i in vital_col_ids]
        features_X_subset_ids = vital_col_ids
    else:
        X_n = len(X_feature_names)
        # X_n = 400
        features_X_subset_ids = list(set(range(X_n)).union(set(vital_col_ids)))

    # Loading np arrays
    features_X = sparse.load_npz(data_dir / "features" / outcome / "X.npz")[
        :n, :, features_X_subset_ids
    ].todense()
    features_s = sparse.load_npz(data_dir / "features" / outcome / "s.npz")[
        :n
    ].todense()

    df_outcome = pd.read_csv(data_dir / "population" / f"{outcome}.csv")[:n]
    df_outcome["y_true"] = df_outcome[f"{outcome.split('_')[0]}_LABEL"]

    return features_X, features_s, X_feature_names, s_feature_names, df_outcome


def get_modes(x, feat, seq_dim=1):
    """
    For a tensor of shape NxLxF
    Returns modal value for given feature across sequence dim.
    """
    # JA: Check conversion to tnsor, dtype etc
    return torch.LongTensor(x[:, :, feat]).mode(dim=seq_dim)[0].clone().detach().numpy()


def split_tasks_fiddle(data, demo, outcome, order="random", seed=SEED):
    """
    Takes FIDDLE format data and given an outcome and demographic,
    splits the input data across that demographic into multiple
    tasks/experiences.
    """
    if demo == "ethnicity_coarse":
        (
            features_X,
            features_s,
            X_feature_names,
            s_feature_names,
            df_outcome,
        ) = get_ethnicity_coarse(data, outcome)
    else:
        (
            features_X,
            features_s,
            X_feature_names,
            s_feature_names,
            df_outcome,
        ) = load_fiddle(data, outcome)

    static_onehot_demos = [
        "sex",
        "age",
        "ethnicity",
        "ethnicity_coarse",
        "hospital",
        "ward",
    ]

    if demo in static_onehot_demos:
        cols = [
            c for c in s_feature_names if c.startswith(DEMO_COL_PREFIXES[data][demo])
        ]
        demo_onehots = [s_feature_names.index(col) for col in cols]
        tasks_idx = [features_s[:, i] == 1 for i in demo_onehots]
    elif demo == "time_season":
        seasons = recover_admission_time(data, outcome)["quarter"]
        tasks_idx = [seasons == i for i in range(1, 5)]
    else:
        raise NotImplementedError

    all_features = concat_timevar_static_feats(features_X, features_s)

    # Reproducible RNG
    if order == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(tasks_idx)
    elif order == "reverse":
        tasks_idx = reversed(tasks_idx)

    tasks = [(all_features[idx], df_outcome[idx]) for idx in tasks_idx]

    return tasks


def concat_timevar_static_feats(features_X, features_s):
    """
    Concatenates time-varying features with static features.
    Static features padded to length of sequence,
    and appended along feature axis.
    """

    # JA: Need to test this has no bugs.

    # Repeat static vals length of sequence across new axis
    s_expanded = np.expand_dims(features_s, 1).repeat(features_X.shape[1], axis=1)
    # Concatenate across feat axis
    all_feats = np.concatenate((features_X, s_expanded), -1)

    return all_feats


def split_trainvaltest_fiddle(
    tasks, val_as_test=True, print_task_partitions=True, seed=SEED
):
    """
    Takes a dataset of multiple tasks/experiences and splits it into train and val/test sets.
    Assumes FIDDLE style outcome/partition cols in df of outcome values.
    """

    # Only MIMIC-III mortality_48h has train/val/test split

    # JA: This currently splits on sample/admission. DOES NOT SPLIT ON PATIENT ID
    # Need to incorporate patient ID split from
    # https://github.com/MLD3/FIDDLE-experiments/blob/master/mimic3_experiments/1_data_extraction/extract_data.py
    # elsewhere defined?

    # Train/val/test/split
    for i in range(len(tasks)):
        if "partition" not in tasks[i][1]:
            # Reproducible RNG
            rng = np.random.default_rng(seed)

            n = len(tasks[i][1])
            partition = rng.choice(["train", "val", "test"], n, p=[0.7, 0.15, 0.15])
            tasks[i][1]["partition"] = partition

    if print_task_partitions:
        partitions = get_task_partition_sizes(tasks)
        for p in partitions:
            print(p)

    if val_as_test:
        tasks_train = [
            (
                t[0][t[1]["partition"] == "train"],
                t[1][t[1]["partition"] == "train"]["y_true"].values,
            )
            for t in tasks
        ]
        tasks_test = [
            (
                t[0][t[1]["partition"] == "val"],
                t[1][t[1]["partition"] == "val"]["y_true"].values,
            )
            for t in tasks
        ]
    else:
        tasks_train = [
            (
                t[0][t[1]["partition"].isin(["train", "val"])],
                t[1][t[1]["partition"].isin(["train", "val"])]["y_true"].values,
            )
            for t in tasks
        ]
        tasks_test = [
            (
                t[0][t[1]["partition"] == "test"],
                t[1][t[1]["partition"] == "test"]["y_true"].values,
            )
            for t in tasks
        ]

    return tasks_train, tasks_test


#############################
# Helper funcs for figs, data, info for paper
#############################


def get_corr_feats_target(df, target):
    cols = df.columns.drop(target)
    df[cols].corr()[target][:]


def get_demo_labels(data, demo, outcome):
    """
    Gets labels for demo splits from feature col names.
    """
    data_dir = DATA_DIR / f"FIDDLE_{data}"

    with open(
        data_dir / "features" / outcome / "s.feature_names.json", encoding="utf-8"
    ) as s_file:
        s_feature_names = json.load(s_file)

    cols = [
        col.split(":")[-1]
        for col in s_feature_names
        if col.startswith(DEMO_COL_PREFIXES[data][demo])
    ]

    return cols


def get_demo_labels_table(demo, datasets=["mimic3", "eicu"]):
    # pd.options.display.max_colwidth = 1000

    # Domainshifts present (over outcomes)
    task_data = []
    outcomes = ["mortality_48h", "ARF_4h", "Shock_4h", "ARF_12h", "Shock_12h"]
    all_tasks = set.union(
        *[
            set(get_demo_labels(data, demo, outcome))
            for outcome in outcomes
            for data in datasets
        ]
    )
    cols = ["Dataset", "Outcome"] + list(all_tasks)

    for data in datasets:
        for outcome in outcomes:
            tasks = get_demo_labels(data, demo, outcome)
            task_data.append(
                [data, outcome.replace("_", " ")]
                + ["\checkmark" if task in tasks else " " for task in all_tasks]
            )

    df = pd.DataFrame(columns=cols, data=task_data)
    df = df.set_index(["Dataset", "Outcome"])

    s = df.sum()
    df = df[s.sort_values(ascending=False).index[:]]

    return df


def get_task_partition_sizes(tasks):
    """
    Prints the number of positive and negative samples in each train/val/test split
    for each task.
    """

    tables = []

    for t in tasks:
        tables.append(
            t[1][["partition", "y_true"]]
            .groupby("partition")
            .agg(Total=("y_true", "count"), Outcome=("y_true", "sum"))
        )
    return tables


def generate_data_tables(data, demo, outcome, seed=SEED):
    """Generate latex tables describing data."""

    tasks = split_tasks_fiddle(data, demo, outcome)

    for i in range(len(tasks)):
        if "partition" not in tasks[i][1]:
            # Reproducible RNG
            rng = np.random.default_rng(seed)

            n = len(tasks[i][1])
            partition = rng.choice(["train", "val", "test"], n, p=[0.7, 0.15, 0.15])
            tasks[i][1]["partition"] = partition

    dfs = get_task_partition_sizes(tasks)

    for i, df in enumerate(dfs):
        df["task"] = i

    df = pd.concat(dfs)
    df = df.set_index(["task"], append=True)
    df = df.unstack()
    df = df.reorder_levels([-1, -2], axis=1)
    df = df.sort_index(axis=1, level=0)

    df = df.reindex(columns=df.columns.reindex(["Total", "Outcome"], level=1)[0])

    return df
