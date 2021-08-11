from utils import data_processing
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def stack_results(results):

    acc = defaultdict(list)

    # Get accuracies for each test set per training "experience"
    for k,v in results.items():
        if 'Top1_Acc_Exp' in k: #and '/Exp' in k:
            new_k = k.split('/')[-1].replace('Exp00','Task ').replace('Exp0','Task ')
            acc[new_k] = v[1]

    df = pd.DataFrame.from_dict(acc)
    df.index.rename(f'Epoch', inplace=True)
    stacked = df.stack().reset_index()
    stacked.rename(columns={'level_1': 'Task', 0: 'Accuracy'}, inplace=True)

    return stacked

def plot_accuracy(method, model, results, ax=None):
    ax = ax or plt.gca()

    stacked = stack_results(results)

    # Only plot task accuracies after examples have been encountered
    #stacked = stacked[stacked['Task'].astype(int) <= stacked['Epoch \n (15 epochs per task)'].astype(int)]

    sns.lineplot(data=stacked, x=f'Epoch', y='Accuracy', hue='Task', ax=ax)
    ax.set_title(method, size=10)
    ax.set_ylabel(model)
    ax.set_xlabel('')

def clean_subplot(i, j, axes):
    ax = axes[i,j]
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if i>0:
        ax.set_title('')
    if i>0 or j>0:
        ax.get_legend().remove()

    plt.setp(axes, ylim=(0,1))

def clean_plot(fig, axes):
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            clean_subplot(i,j,axes)
            
    handles, labels = axes[0,0].get_legend_handles_labels()
    axes[0,0].get_legend().remove()
    fig.legend(handles, labels, loc='center right', title='Task')

    fig.suptitle('Continual Learning model comparison')
    fig.supxlabel('Epoch')
    fig.supylabel('Accuracy', x=0)

def plot_demos():
    """
    Plots demographic information of eICU dataset.
    """

    df = data_processing.load_eicu(drop_dupes=True)
    fig, axes = plt.subplots(3,2, sharey=True, figsize=(18,18), squeeze=False)

    df['gender'].value_counts().plot.bar(ax=axes[0,0], rot=0, title='Gender')
    df['ethnicity'].value_counts().plot.bar(ax=axes[1,0], rot=0, title='Ethnicity')
    df['ethnicity_coarse'].value_counts().plot.bar(ax=axes[1,1], rot=0, title='Ethnicity (coarse)')
    df['age'].plot.hist(bins=20, label='age', ax=axes[0,1], title='Age')
    df['region'].value_counts().plot.bar(ax=axes[2,0], rot=0, title='Region (North America)')
    df['hospitaldischargestatus'].value_counts().plot.bar(ax=axes[2,1], rot=0, title='Outcome')
    plt.show()
    plt.close()
