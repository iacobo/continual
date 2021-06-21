#%%
####################
# PLOTTING
####################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch import nn
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from avalanche.training.strategies import Naive, JointTraining, Cumulative #Baselines
from avalanche.training.strategies import EWC, LwF, SynapticIntelligence   #Regularisation
from avalanche.training.strategies import Replay, GEM, AGEM                #Rehearsal
#from avalanche.training.strategies import AR1, CWRStar, GDumb, CoPE, StreamingLDA #???
from avalanche.benchmarks.generators import tensors_benchmark, ni_benchmark

from models import SimpleMLP, SimpleCNN, SimpleLSTM, SimpleRNN

def plot_accuracy(method, model, results, ax=None):
    ax = ax or plt.gca()

    acc = defaultdict(list)

    for result in results:
        for k,v in result.items():
            if 'Top1_Acc_Exp' in k and '/Exp' in k:
                new_k = k.split('/')[-1].replace('Exp00','').replace('Exp0','')
                acc[new_k].append(v)

    df = pd.DataFrame.from_dict(acc, orient='index')
    df.index.rename(f'Task introduced \n ({train_epochs} epochs per task)', inplace=True)
    stacked = df.stack().reset_index()
    stacked.rename(columns={'level_1': 'Task', 0: 'Accuracy'}, inplace=True)

    # Only plot task accuracies after examples have been encountered
    #stacked = stacked[stacked['Task'].astype(int) <= stacked['Task introduced \n (15 epochs per task)'].astype(int)]

    sns.lineplot(data=stacked, x=f'Task introduced \n ({train_epochs} epochs per task)', y='Accuracy', hue='Task', ax=ax)
    ax.title.set_text(method)
    ax.set_ylabel(model)

def clean_plot(i, j, ax=None):
    ax = ax or plt.gca()
    if i!=0:
        #ax.xaxis.label.set_visible(False)
        ax.set_title('')
    if j!=0:
        ax.yaxis.label.set_visible(False)
    if i!=0 or j!=0:
        ax.get_legend().remove()


######################
# DATA LOADING
######################
# Loading data into 'stream' of 'experiences' (tasks)

n_timesteps = 5
n_channels = 2
n_tasks = 5

# Definition of training experiences
# of form (x:(patients, variables, time_steps), y:(outcome))

from data_processing import generate_experiences

experiences = generate_experiences(n_timesteps, n_channels, n_tasks)
test_experiences = generate_experiences(n_timesteps, n_channels, n_tasks, test=True)

scenario = tensors_benchmark(
    train_tensors=experiences,
    test_tensors=test_experiences,
    task_labels=[0 for _ in range(n_tasks)],  # Task label of each train exp
    complete_test_set_only=False
)

#from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset

#scenario = ni_benchmark(
#    train_dataset=[(experience_1_x, experience_1_y), (experience_2_x, experience_2_y)],
#    test_dataset=[(test_1_x, test_1_y), (test_2_x, test_2_y)],
#    n_experiences=2,
#    task_labels=[0, 0],  # Task label of each train exp
#)

print('Simul Data loaded!')

#####################
# DEFINE MODEL
#####################

cnn = SimpleCNN(n_channels=n_channels)
mlp = SimpleMLP(n_channels=n_channels, seq_len=n_timesteps)
rnn = SimpleRNN(n_channels=n_channels, seq_len=n_timesteps)
lstm = SimpleLSTM(n_channels=n_channels, seq_len=n_timesteps)
models = {'MLP':mlp, 'CNN':cnn, 'RNN':rnn, 'LSTM:':lstm}

print('Models defined!')

######################
# Define CL strategy
######################

def load_strategy(model, strategy, **kwargs):
    """
    """
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    # Params
    train_mb_size=100
    train_epochs=6
    eval_mb_size=100

    strategies = {'Naive':Naive, 'Joint':JointTraining, 'Cumulative':Cumulative,
    'EWC':EWC, 'LwF':LwF, 'SI':SynapticIntelligence, 'Replay':Replay, 
    'GEM':GEM, 'AGEM':AGEM}

    strategy = strategies[strategy]

    model = strategy(
        model, optimizer, criterion, 
        train_mb_size=train_mb_size, train_epochs=train_epochs, eval_mb_size=eval_mb_size,
        **kwargs
    )

    return model


#####################
# TRAINING
#####################

# TRAINING LOOP
def train_method(cl_strategy, scenario):
    """
    """
    print('Starting experiment...')
    results = []
    for i, experience in enumerate(scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        #print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(scenario.test_stream))

    return results

# Need to rerun multiple times, take averages, shuffle each task data internally
kwargs = {'Naive':{}, 'Joint':{}, 'Replay':{'mem_size':10}, 'EWC':{'ewc_lambda':0.01}}
strategies = kwargs.keys()
#models = [model] # Need new instance of each model for each training loop or remembers weights
fig, axes = plt.subplots(len(models), len(strategies))

train_epochs=6

for i, (model_name, model) in enumerate(models.items()):
    # Save initial weights for iterating over methods
    sd = model.state_dict()
    for j, strategy in enumerate(strategies):
        # Reload initial weights for each method
        model.load_state_dict(sd)
        cl_strategy = load_strategy(model, strategy, **kwargs[strategy])
        results = train_method(cl_strategy, scenario)
        plot_accuracy(strategy,model_name, results, axes[i,j])
        clean_plot(i, j, axes[i,j])

axes[0,0].get_shared_y_axes().join(*[sp for ax in axes for sp in ax])
handles, labels = axes[0,0].get_legend_handles_labels()
axes[0,0].get_legend().remove()
fig.legend(handles, labels, loc='center right')

fig.supxlabel(' \n\n\n\nEpoch')
fig.supylabel('Accuracy\n\n\n\n')

plt.show()
