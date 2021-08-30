"""
Functions for plotting results and descriptive analysis of data.
"""

from datetime import datetime
from collections import defaultdict

import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#from utils import data_processing

METRIC_FULL_NAME = {'BalAcc': 'Balanced Accuracy',
                    'Top1_Acc': 'Accuracy',
                    'Loss': 'Loss'}

def get_timestamp():
    """
    Returns current timestamp as string.
    """
    ts = time.time()
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

def stack_results(results, metric):
    """
    Stacks results
    """
    metric_dict = defaultdict(list)

    # Get metrics for each training "experience"'s test set
    for k,v in results.items():
        if f'{metric}_Exp' in k:
            new_k = k.split('/')[-1].replace('Exp00','Task ').replace('Exp0','Task ')
            metric_dict[new_k] = v[1]

    df = pd.DataFrame.from_dict(metric_dict)
    df.index.rename('Epoch', inplace=True)
    stacked = df.stack().reset_index()
    stacked.rename(columns={'level_1': 'Task', 0: METRIC_FULL_NAME[metric]}, inplace=True)

    return stacked

def plot_metric(method, model, results, metric, ax=None):
    """
    Plots given metric from dict.
    Stacks multiple plots (i.e. different per-task metrics) over training time.
    """
    ax = ax or plt.gca()

    stacked = stack_results(results, metric)

    # Only plot task accuracies after examples have been encountered
    #stacked = stacked[stacked['Task'].astype(int)<=stacked['Epoch \n (15 epochs per task)'].astype(int)]

    sns.lineplot(data=stacked, x='Epoch', y=METRIC_FULL_NAME[metric], hue='Task', ax=ax)
    ax.set_title(method, size=10)
    ax.set_ylabel(model)
    ax.set_xlabel('')

def clean_subplot(i, j, axes):
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

    plt.setp(axes, ylim=(0,1))

def clean_plot(fig, axes):
    """
    Cleans all subpots. Removes duplicate legends.
    """
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            clean_subplot(i,j,axes)
            
    handles, labels = axes[0,0].get_legend_handles_labels()
    axes[0,0].get_legend().remove()
    fig.legend(handles, labels, loc='center right', title='Task')

def annotate_plot(fig, demo, metric, outcome='48h mortality'):
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
                 f'Outcome: {outcome} | Domain Increment: {demo}')

def plot_all_model_strats(models, strategies, data, demo, res, results_dir, savefig=True, metric='BalAcc'):
    """
    Pairplot of all models vs strategies.
    """
    fig, axes = plt.subplots(len(models), len(strategies), sharex=True, sharey=True, figsize=(8,8*(len(models)/len(strategies))), squeeze=False)

    for i, model in enumerate(models):
        for j, strategy in enumerate(strategies):
            plot_metric(strategy, model, res[model][strategy], metric, axes[i,j])

    clean_plot(fig, axes)
    annotate_plot(fig, demo, metric)

    if savefig:
        plt.savefig(results_dir / 'figs' / f'fig_{data}_{demo}_{metric}_{get_timestamp()}.png')

def plot_demos():
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