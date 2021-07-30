import copy
import time
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt

# ML imports
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

# CL imports
from avalanche.training.strategies import Naive, JointTraining, Cumulative # Baselines
from avalanche.training.strategies import EWC, LwF, SynapticIntelligence   # Regularisation
from avalanche.training.strategies import Replay, GDumb, GEM, AGEM         # Rehearsal
from avalanche.training.strategies import AR1, CWRStar, CoPE, StreamingLDA # Misc

from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.benchmarks.generators import tensors_benchmark

# Local imports
from models import SimpleMLP, SimpleCNN, SimpleLSTM, SimpleRNN
from plotting import plot_accuracy, clean_plot
from data_processing import eicu_to_tensor, random_data


# HELPER FUNCTIONS

def load_strategy(model, model_name, strategy_name, train_epochs=20, eval_every=1, train_mb_size=128, eval_mb_size=1024, weight=None, output_dir=Path(''), timestamp='', **kwargs):
    """
    """
    try:
        optimizer = kwargs['optimizer'](model.parameters(), lr=kwargs['lr'], momentum=0.9)
    except TypeError:
        optimizer = kwargs['optimizer'](model.parameters(), lr=kwargs['lr'])

    criterion = CrossEntropyLoss(weight=weight)

    strategies = {'Naive':Naive, 'Joint':JointTraining, 'Cumulative':Cumulative,
    'EWC':EWC, 'LwF':LwF, 'SI':SynapticIntelligence, 'Replay':Replay, 
    'GEM':GEM, 'AGEM':AGEM, 'GDumb':GDumb, 'CoPE':CoPE, 
    'AR1':AR1, 'StreamingLDA':StreamingLDA, 'CWRStar':CWRStar}

    strategy = strategies[strategy_name.split('_')[0]]

    # Loggers
    interactive_logger = InteractiveLogger()
    text_logger = TextLogger(open(output_dir / 'log.txt', 'a'))
    tb_logger = TensorboardLogger(tb_log_dir = output_dir / f'tb_data_{timestamp}' / model_name / strategy_name) # JA ROOT

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True, experience=True), #Stream for avg accuracy over all over all experiences, experience=True for individuals (Former may rely on tasks being roughly same size?)
        loggers=[interactive_logger, tb_logger, text_logger])

    if strategy_name == 'StreamingLDA':
        del kwargs['optimizer']

        if model_name == 'CNN':
            kwargs['input_size'] = (30//4)*(512//4)
            kwargs['output_layer_name'] = 'cnn_layers'

    else:
        kwargs['optimizer'] = optimizer
        del kwargs['lr']

    model = strategy(
        model, #optimizer=optimizer, 
        criterion=criterion,
        train_mb_size=train_mb_size, eval_mb_size=eval_mb_size,
        train_epochs=train_epochs, eval_every=eval_every, evaluator=eval_plugin,
        **kwargs
    )

    return model

def train_method(cl_strategy, scenario, eval_on_test=True):
    """
    """
    print('Starting experiment...')

    if eval_on_test:
        eval_streams=[scenario.test_stream]
    else:
        eval_streams=[scenario.train_stream]

    for experience in scenario.train_stream:
        print(f'Start of experience: {experience.current_experience}')
        cl_strategy.train(experience, eval_streams=eval_streams)
        print('Training completed', '\n\n')

    results = cl_strategy.evaluator.get_all_metrics()

    return results

# CL hyper-params
# https://arxiv.org/pdf/2103.07492.pdf


def main(data='random', demo='region', models=['MLP'], output_dir=Path('.'), config=None):

    """
    data: ['random','MIMIC','eICU','iORD']
    demo: ['region','sex','age','ethnicity','ethnicity_coarse','hospital']
    """

    # Timestamp for logging
    ts = time.time()
    timestamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    data_dir = output_dir / 'data' / data

    # Loading data into 'stream' of 'experiences' (tasks)
    # of form (x:(samples, variables, time_steps), y:(outcome,))
    print('Loading data...')
    if data=='eICU':
        experiences = eicu_to_tensor(demographic=demo, balance=True, root=data_dir)
        test_experiences = copy.deepcopy(experiences)

    elif data=='random':
        experiences = random_data()
        test_experiences = copy.deepcopy(experiences)

    elif data=='MIMIC':
        raise NotImplemented
    
    elif data=='iORD':
        raise NotImplemented

    else:
        print('Unknown data source.')
        pass

    print('Data loaded.')

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

    # Define models
    all_models = {'MLP':SimpleMLP, 'CNN':SimpleCNN, 'RNN':SimpleRNN, 'LSTM':SimpleLSTM}
    models = {k: all_models[k] for k in models}

    # Define CL strategy
    strategies = {}
    #strategies = {'Naive':{}, 'Cumulative':{}, 'Replay':{'mem_size':50}} #, 'SI_1':{'si_lambda':1}, 'LwF_1':{'alpha':1, 'temperature':0.5}, 'EWC':{'ewc_lambda':100}} #, 'StreamingLDA':{'input_size':int(0.4*512), 'num_classes':2, 'output_layer_name':'features'}}

    # Hyperparam gridsearch:
    for lr in config['lr']:
        for opt_name, optim in config['optimizer'].items():
            strategies[f'Naive_{lr}_{opt_name}'.replace('.','_')] = {'lr':lr, 'optimizer':optim}
            strategies[f'Cumulative_{lr}_{opt_name}'.replace('.','_')] = {'lr':lr, 'optimizer':optim}

            for mem_size in config['mem_size']:
                strategies[f'Replay_{lr}_{opt_name}_{mem_size}'.replace('.','_')] = {'lr':lr, 'optimizer':optim, 'mem_size':mem_size}

            # Loop over CL specific hyperparams
            for alpha in config['alpha']:
                strategies[f'SI_{lr}_{opt_name}_{alpha}'.replace('.','_')] = {'si_lambda':alpha, 'lr':lr, 'optimizer':optim}
                strategies[f'EWC_{lr}_{opt_name}_{alpha}'.replace('.','_')] = {'ewc_lambda':alpha, 'lr':lr, 'optimizer':optim}

                for temp in config['temperature']:
                    strategies[f'LwF_{lr}_{opt_name}_{alpha}_{temp}'.replace('.','_')] = {'alpha':alpha, 'temperature':temp, 'lr':lr, 'optimizer':optim}
    
    # TRAINING (Need to rerun multiple times, take averages)
    # Container for metrics for plotting
    res = {m:{s:None for s in strategies.keys()} for m in models.keys()}

    for model_name in models.keys():
        for strategy_name, kwargs in strategies.items():

            # Union model and strategy config's and pass to raytune
            # one ray per model/strat combo

            # Secondary experiment: how sensitive reg strats are to hyperparams

            # This is tune train func
            model = models[model_name](n_channels=n_channels, seq_len=n_timesteps)
            cl_strategy = load_strategy(model, model_name, strategy_name, weight=None, output_dir=output_dir, timestamp=timestamp, **kwargs)
            results = train_method(cl_strategy, scenario, eval_on_test=False)
            res[model_name][strategy_name] = results

            # tune.report(loss=(val_loss / val_steps), accuracy=correct / total)


    # PLOTTING
    fig, axes = plt.subplots(len(models), len(strategies), sharex=True, sharey=True, figsize=(8,8*(len(models)/len(strategies))), squeeze=False)

    for i, model in enumerate(models.keys()):
        for j, strategy in enumerate(strategies.keys()):
            plot_accuracy(strategy, model, res[model][strategy], axes[i,j])

    clean_plot(fig, axes)
    plt.savefig(output_dir / 'figs' / f'fig_{timestamp}.png')
    #plt.show()

    return res