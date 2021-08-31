"""
Contains functions for running hyperparameter sweep and
Continual Learning model-training and evaluation.
"""

from pathlib import Path
from functools import partial

import json
import warnings
import torch
from ray import tune
from torch import nn, optim

from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, StreamConfusionMatrix

# Local imports
from utils import models, plotting, data_processing, cl_strategies
from utils.metrics import balancedaccuracy_metrics

# Suppressing erroneous MaxPool1d named tensors warning
warnings.filterwarnings("ignore", category=UserWarning)

# GLOBALS
RESULTS_DIR = Path(__file__).parents[1] / 'results'
CONFIG_DIR = Path(__file__).parents[1] / 'config'
CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA else 'cpu'

def save_params(data, demo, model, strategy, best_params):
    """
    Save hyper-param config to json.
    """
    with open(CONFIG_DIR / f'config_{data}_{demo}_{model}_{strategy}.json', 'w', encoding='utf-8') as json_file:
        json.dump(best_params, json_file)

def load_params(data, demo, model, strategy):
    """
    Load hyper-param config from json.
    """
    with open(CONFIG_DIR / f'config_{data}_{demo}_{model}_{strategy}.json', encoding='utf-8') as json_file:
        best_params = json.load(json_file)
    return best_params

def load_strategy(model, model_name, strategy_name, data='', demo='', weight=None, validate=False, config=None, benchmark=None):
    """
    - `stream`     Avg accuracy over all experiences (may rely on tasks being roughly same size?)
    - `experience` Accuracy for each experience
    """

    strategy = cl_strategies.STRATEGIES[strategy_name]
    criterion = nn.CrossEntropyLoss(weight=weight)

    if config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    elif config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Loggers
    # JA: subfolders for datset / experiments VVV

    if validate:
        loggers = []
    else:
        timestamp = plotting.get_timestamp()
        log_dir = RESULTS_DIR / 'log' / 'tensorboard' / f'{data}_{demo}_{timestamp}' / model_name / strategy_name
        interactive_logger = InteractiveLogger()
        tb_logger = TensorboardLogger(tb_log_dir=log_dir)
        loggers = [interactive_logger, tb_logger]

    eval_plugin = EvaluationPlugin(
        StreamConfusionMatrix(save_image=False),
        loss_metrics(stream=True, experience=not validate),
        accuracy_metrics(stream=True, experience=not validate),
        balancedaccuracy_metrics(stream=True, experience=not validate),
        loggers=loggers,
        benchmark=benchmark)

    cl_strategy = strategy(
        model,
        optimizer=optimizer,
        device=DEVICE,
        criterion=criterion,
        eval_mb_size=1024,
        eval_every=-1 if validate else 1,
        evaluator=eval_plugin,
        train_epochs=config['train_epochs'],
        train_mb_size=config['train_mb_size'],
        **config['strategy']
    )

    return cl_strategy

def train_cl_method(cl_strategy, scenario, validate=False):
    """
    Avalanche Cl training loop. For each 'experience' in scenario's train_stream:

        - Trains method on experience
        - evaluates model on test_stream
    """
    print('Starting experiment...')

    for experience in scenario.train_stream:
        print(f'Start of experience: {experience.current_experience}')
        cl_strategy.train(experience, eval_streams=[scenario.train_stream, scenario.test_stream])
        print('Training completed', '\n\n')

    if validate:
        return cl_strategy.eval(scenario.test_stream)
    else:
        return cl_strategy.evaluator.get_all_metrics()

def training_loop(config, data, demo, model_name, strategy_name, validate=False, checkpoint_dir=None):
    """
    Training wrapper:
        - loads data
        - instantiates model
        - equips model with CL strategy
        - trains and evaluates method
        - returns either results or hyperparam optimisation if `validate`
    """

    # Loading data into 'stream' of 'experiences' (tasks)
    print('Loading data...')
    scenario, _, n_timesteps, n_channels, weight = data_processing.load_data(data, demo, validate)
    if weight is not None:
        weight = weight.to(DEVICE)
    print('Data loaded.')
    print(f'N timesteps: {n_timesteps}\n'
          f'N features:  {n_channels}')

    model = models.MODELS[model_name](n_channels, n_timesteps, config['hidden_dim'], **config['model'])
    cl_strategy = load_strategy(model, model_name, strategy_name, data, demo, weight=weight, validate=validate, config=config, benchmark=scenario)
    results = train_cl_method(cl_strategy, scenario, validate=validate)

    # Garbage collection
    del cl_strategy
    del model
    torch.cuda.empty_cache()

    if validate:
        loss = results['Loss_Stream/eval_phase/test_stream/Task000']
        accuracy = results['Top1_Acc_Stream/eval_phase/test_stream/Task000']
        balancedaccuracy = results['BalAcc_Stream/eval_phase/test_stream/Task000']

        # WARNING: `return` overwrites raytune report
        tune.report(loss=loss, accuracy=accuracy, balancedaccuracy=balancedaccuracy)

    else:
        return results

def hyperparam_opt(config, data, demo, model_name, strategy_name, num_samples=5):
    """
    Hyperparameter optimisation for the given model/strategy.
    Runs over the validation data for the first 2 tasks.

    Can use returned optimal values to later run full training and testing over all n>=2 tasks.
    """

    reporter = tune.CLIReporter(metric_columns=['loss', 'accuracy', 'balancedaccuracy'])
    resources = {'cpu':4, 'gpu':1} if CUDA else {'cpu':1}

    result = tune.run(
        partial(training_loop,
                data=data,
                demo=demo,
                model_name=model_name,
                strategy_name=strategy_name,
                validate=True),
        config=config,
        num_samples=num_samples,
        progress_reporter=reporter,
        raise_on_failed_trial=False,
        resources_per_trial=resources,
        name=f'{model_name}_{strategy_name}',
        local_dir=RESULTS_DIR / 'log' / 'raytune' / f'{data}_{demo}',
        trial_name_creator=lambda t: f'{model_name}_{strategy_name}_{t.trial_id}')

    best_trial = result.get_best_trial('balancedaccuracy', 'max', 'last')
    print(f'Best trial config:                             {best_trial.config}')
    print(f'Best trial final validation loss:              {best_trial.last_result["loss"]}')
    print(f'Best trial final validation accuracy:          {best_trial.last_result["accuracy"]}')
    print(f'Best trial final validation balanced accuracy: {best_trial.last_result["balancedaccuracy"]}')

    return best_trial.config

def main(data='random', demo='', models=['MLP'], strategies=['Naive'], 
         config_generic={}, config_model={}, config_cl={}, 
         validate=False, num_samples=5):
    """
    Main training loop. Takes dataset, demographic splits, 
    and evaluates model/strategies over given hyperparams over this problem.
    """

    # TRAINING
    # Container for metrics for plotting
    res = {m:{s:None for s in strategies} for m in models}

    for model in models:
        for strategy in strategies:
            # Hyperparam opt over first 2 tasks
            if validate:
                config = {**config_generic, 'model':config_model[model], 'strategy':config_cl.get(strategy,{})}
                best_params = hyperparam_opt(config, data, demo, model, strategy, num_samples=num_samples)
                save_params(data, demo, model, strategy, best_params)
            # Training loop over all tasks
            # JA: (Need to rerun multiple times for mean + CI's)
            # for i in range(5): res[model][strategy].append(...)
            else:
                config = load_params(data, demo, model, strategy)
                res[model][strategy] = training_loop(config, data, demo, model, strategy)

    # PLOTTING
    if not validate:
        # Locally saving results
        with open(RESULTS_DIR / f'results_{data}_{demo}.json', 'w', encoding='utf-8') as handle:
            res_no_tensors = {m:{s:{metric:value for metric, value in metrics.items() if 'Confusion' not in metric}
                                                 for s, metrics in strats.items()} 
                                                 for m, strats in res.items()}
            json.dump(res_no_tensors, handle)

        plotting.plot_all_model_strats(models, strategies, data, demo, res, results_dir=RESULTS_DIR, savefig=True)
