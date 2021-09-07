"""
Functions for plotting results and descriptive analysis of data.
"""

from pathlib import Path
from datetime import datetime
from collections import defaultdict

import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).parents[1] / 'results'

METRIC_FULL_NAME = {
    'Top1_Acc': 'Accuracy',
    'BalAcc': 'Balanced Accuracy',
    'Loss': 'Loss'
    }

def get_timestamp():
    """
    Returns current timestamp as string.
    """
    ts = time.time()
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

def stack_results(results, metric, mode):
    """
    Stacks results
    """
    metric_dict = defaultdict(list)

    # Get metrics for each training "experience"'s test set
    for k,v in results.items():
        if f'{metric}_Exp/eval_phase/{mode}_stream' in k:
            new_k = k.split('/')[-1].replace('Exp00','Task ').replace('Exp0','Task ')
            metric_dict[new_k] = v[1]

    df = pd.DataFrame.from_dict(metric_dict)
    df.index.rename('Epoch', inplace=True)
    stacked = df.stack().reset_index()
    stacked.rename(columns={'level_1': 'Task', 0: METRIC_FULL_NAME[metric]}, inplace=True)

    return stacked

def plot_metric(method, model, results, mode, metric, ax=None):
    """
    Plots given metric from dict.
    Stacks multiple plots (i.e. different per-task metrics) over training time.

    `mode`: ['train','test'] (which stream to plot)
    """
    ax = ax or plt.gca()

    stacked = stack_results(results, metric, mode)

    # Only plot task accuracies after examples have been encountered
    #stacked = stacked[stacked['Task'].astype(int)<=stacked['Epoch \n (15 epochs per task)'].astype(int)]

    sns.lineplot(data=stacked, x='Epoch', y=METRIC_FULL_NAME[metric], hue='Task', ax=ax)
    ax.set_title(method, size=10)
    ax.set_ylabel(model)
    ax.set_xlabel('')

def clean_subplot(i, j, axes, metric):
    """
    Removes top/rights spines.
    Removes titles/legend.
    Fixes y/metric limits.
    """
    ax = axes[i,j]
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if i>0:
        ax.set_title('')
    if i>0 or j>0:
        ax.get_legend().remove()

    if metric != 'Loss':
        plt.setp(axes, ylim=(0,1))

def clean_plot(fig, axes, metric):
    """
    Cleans all subpots. Removes duplicate legends.
    """
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            clean_subplot(i,j,axes,metric)
            
    handles, labels = axes[0,0].get_legend_handles_labels()
    axes[0,0].get_legend().remove()
    fig.legend(handles, labels, loc='center right', title='Task')

def annotate_plot(fig, domain, outcome, metric):
    """
    Adds x/y labels and suptitles.
    """
    try:
        fig.supxlabel('Epoch')
        fig.supylabel(METRIC_FULL_NAME[metric], x=0)
    except AttributeError:
        fig.text(0.5, 0.04, 'Epoch', ha='center')
        fig.text(0.04, 0.5, METRIC_FULL_NAME[metric], va='center', rotation='vertical')

    fig.suptitle(f'Continual Learning model comparison \n'
                 f'Outcome: {outcome} | Domain Increment: {domain}')

def plot_all_model_strats(models, strategies, data, domain, outcome, res, mode, metric, savefig=True):
    """
    Pairplot of all models vs strategies.
    """

    # Loss can blow up orders of magnitude in some experiments
    sharey = False if metric=='Loss' else True

    fig, axes = plt.subplots(len(models), len(strategies), sharex=True, sharey=sharey, figsize=(8,8*(len(models)/len(strategies))), squeeze=False)

    for i, model in enumerate(models):
        for j, strategy in enumerate(strategies):
            plot_metric(strategy, model, res[model][strategy], mode, metric, axes[i,j])

    clean_plot(fig, axes, metric)
    annotate_plot(fig, domain, outcome, metric)

    if savefig:
        plt.savefig(RESULTS_DIR / 'figs' / f'fig_{data}_{domain}_{mode}_{metric}_{get_timestamp()}.png')

def plot_demographics():
    """
    Plots demographic information of eICU dataset.
    """

    df = pd.DataFrame() #data_processing.load_eicu(drop_dupes=True)
    _, axes = plt.subplots(3,2, sharey=True, figsize=(18,18), squeeze=False)

    df['gender'].value_counts().plot.bar(ax=axes[0,0], rot=0, title='Gender')
    df['ethnicity'].value_counts().plot.bar(ax=axes[1,0], rot=0, title='Ethnicity')
    df['ethnicity_coarse'].value_counts().plot.bar(ax=axes[1,1], rot=0, title='Ethnicity (coarse)')
    df['age'].plot.hist(bins=20, label='age', ax=axes[0,1], title='Age')
    df['region'].value_counts().plot.bar(ax=axes[2,0], rot=0, title='Region (North America)')
    df['hospitaldischargestatus'].value_counts().plot.bar(ax=axes[2,1], rot=0, title='Outcome')
    plt.show()
    plt.close()