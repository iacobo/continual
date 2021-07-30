import copy
import time
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
from functools import partial

# ML imports
from torch.nn import CrossEntropyLoss
from ray import tune
from ray.tune import CLIReporter

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

def load_strategy(model, model_name, strategy_name, train_epochs=20, eval_every=1, train_mb_size=128, eval_mb_size=1024, weight=None, output_dir=Path(''), timestamp='', **config):
    """
    """
    try:
        optimizer = config['optimizer'](model.parameters(), lr=config['lr'], momentum=0.9)
    except TypeError:
        optimizer = config['optimizer'](model.parameters(), lr=config['lr'])

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

    model = strategy(
        model, optimizer=optimizer, 
        criterion=criterion,
        train_mb_size=train_mb_size, eval_mb_size=eval_mb_size,
        train_epochs=train_epochs, eval_every=eval_every, evaluator=eval_plugin,
        **{k:v for k, v in config.items() if k not in ('optimizer','lr')}
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

    if validate:
        return cl_strategy.eval(scenario.test_stream)

    else:
        return results

# CL hyper-params
# https://arxiv.org/pdf/2103.07492.pdf

def training_loop(models, model_name, strategy_name, scenario, output_dir, timestamp, n_channels, n_timesteps, config, validate=False):

    # This is tune train func
    model = models[model_name](n_channels=n_channels, seq_len=n_timesteps)
    cl_strategy = load_strategy(model, model_name, strategy_name, weight=None, output_dir=output_dir, timestamp=timestamp, **config)
    results = train_method(cl_strategy, scenario, eval_on_test=False)

    if validate:
        tune.report(loss=results['Loss'], accuracy=results['Accuracy'])

    return results

def hyperparam_opt(models, model_name, strategy_name, scenario, output_dir, timestamp, n_channels, n_timesteps, config):
    """
    Hyperparameter optimisation for the given model/strategy.
    Runs over the validation data for the first 2 tasks.

    Can use returned optimal values to later run full training and testing over all n>=2 tasks.
    """

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy"])
    result = tune.run(
        partial(training_loop, validate=True, models=models, model_name=model_name, strategy_name=strategy_name, scenario=scenario, output_dir=output_dir, timestamp=timestamp, n_channels=n_channels, n_timesteps=n_timesteps),
        config=config,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f'Best trial config: {best_trial.config}')
    print(f'Best trial final validation loss: {best_trial.last_result["loss"]}')
    print(f'Best trial final validation accuracy: {best_trial.last_result["accuracy"]}')



def main(data='random', demo='region', models=['MLP'], output_dir=Path('.'), config_generic=None, config_cl=None):

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

    elif data=='MIMIC': raise NotImplemented
    elif data=='iORD': raise NotImplemented
    else:
        print('Unknown data source.')
        pass

    print('Data loaded.')

    if validate:
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

    # Define models
    all_models = {'MLP':SimpleMLP, 'CNN':SimpleCNN, 'RNN':SimpleRNN, 'LSTM':SimpleLSTM}
    models = {k: all_models[k] for k in models}

    # Define CL strategy
    strategies = ['Naive', 'Cumulative', 'Replay', 'SI', 'LwF', 'EWC'] 

    # TRAINING (Need to rerun multiple times, take averages)
    # Container for metrics for plotting CHANGE TO TXT FILE
    res = {m:{s:None for s in strategies.keys()} for m in models.keys()}

    for model_name in models.keys():
        for strategy_name in strategies:

            # Union generic and CL strategy-specific hyperparams
            config = {**config_generic, **config_cl[strategy_name]}
            # Training loop
            res[model_name][strategy_name] = training_loop(models, model_name, strategy_name, scenario, output_dir, timestamp, n_channels, n_timesteps, config)

            # Tune best config add

            # Secondary experiment: how sensitive regularization strategies are to hyperparams
            # Tune hyperparams over increasing number of tasks?


    # PLOTTING
    fig, axes = plt.subplots(len(models), len(strategies), sharex=True, sharey=True, figsize=(8,8*(len(models)/len(strategies))), squeeze=False)

    for i, model in enumerate(models.keys()):
        for j, strategy in enumerate(strategies.keys()):
            plot_accuracy(strategy, model, res[model][strategy], axes[i,j])

    clean_plot(fig, axes)
    plt.savefig(output_dir / 'figs' / f'fig_{timestamp}.png')
    #plt.show()

    return res