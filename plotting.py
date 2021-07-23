import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

def stack_results(results):

    acc = defaultdict(list)

    # Get accuracies for each test set per training "experience"
    for k,v in results.items():
        if 'Top1_Acc_Exp' in k and '/Exp' in k:
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
    
    plt.show()