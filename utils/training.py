"""
Contains functions for running hyperparameter sweep and
Continual Learning model-training and evaluation.
"""

import json
import warnings
from pathlib import Path
from functools import partial

import torch
from ray import tune
from torch import nn, optim

from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, StreamConfusionMatrix

# Local imports
from utils import models, plotting, data_processing, cl_strategies
from utils.metrics import balancedaccuracy_metrics, sensitivity_metrics, specificity_metrics, precision_metrics, rocauc_metrics, auprc_metrics

# Suppressing erroneous MaxPool1d named tensors warning
warnings.filterwarnings("once", category=UserWarning)

# GLOBALS
RESULTS_DIR = Path(__file__).parents[1] / 'results'
CONFIG_DIR = Path(__file__).parents[1] / 'config'
CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA else 'cpu'

def save_params(data, domain, outcome, model, strategy, best_params):
    """Save hyper-param config to json."""

    file_loc = CONFIG_DIR / data / outcome / domain
    file_loc.mkdir(parents=True, exist_ok=True)

    with open(file_loc / f'config_{model}_{strategy}.json', 'w', encoding='utf-8') as json_file:
        json.dump(best_params, json_file)

def load_params(data, domain, outcome, model, strategy):
    """Load hyper-param config from json."""

    file_loc = CONFIG_DIR / data / outcome / domain
    
    with open(file_loc / f'config_{model}_{strategy}.json', encoding='utf-8') as json_file:
        best_params = json.load(json_file)
    return best_params

def save_results(data, outcome, domain, res):
    """Saves results to .json (excluding tensor confusion matrix)."""
    with open(RESULTS_DIR / f'results_{data}_{outcome}_{domain}.json', 'w', encoding='utf-8') as handle:
        res_no_tensors = {m:{s:[{metric:value for metric, value in run.items() if 'Confusion' not in metric} for run in runs]
                                                for s, runs in strats.items()} 
                                                for m, strats in res.items()}
        json.dump(res_no_tensors, handle)

def load_strategy(model, model_name, strategy_name, data='', domain='', n_tasks=0, weight=None, validate=False, config=None, benchmark=None):
    """
    - `stream`     Avg accuracy over all experiences (may rely on tasks being roughly same size?)
    - `experience` Accuracy for each experience
    """

    strategy = cl_strategies.STRATEGIES[strategy_name]
    criterion = nn.CrossEntropyLoss(weight=weight)

    if config['generic']['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['generic']['lr'], momentum=0.9)
    elif config['generic']['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['generic']['lr'])

    if validate:
        loggers = []
    else:
        timestamp = plotting.get_timestamp()
        log_dir = RESULTS_DIR / 'log' / 'tensorboard' / f'{data}_{domain}_{timestamp}' / model_name / strategy_name
        interactive_logger = InteractiveLogger()
        tb_logger = TensorboardLogger(tb_log_dir=log_dir)
        loggers = [interactive_logger, tb_logger]

    eval_plugin = EvaluationPlugin(
        StreamConfusionMatrix(save_image=False),
        loss_metrics(stream=True, experience=not validate),
        accuracy_metrics(trained_experience=True, experience=not validate),
        balancedaccuracy_metrics(trained_experience=True, experience=not validate),
        specificity_metrics(trained_experience=True, experience=not validate),
        sensitivity_metrics(trained_experience=True, experience=not validate),
        precision_metrics(trained_experience=True, experience=not validate),
        #rocauc_metrics(trained_experience=True, experience=not validate),
        #auprc_metrics(trained_experience=True, experience=not validate),
        loggers=loggers,
        benchmark=benchmark)

    cl_strategy = strategy(
        model,
        optimizer=optimizer,
        device=DEVICE,
        criterion=criterion,
        eval_mb_size=1024,
        eval_every=0, #if validate or n_tasks > 5 else 1,
        evaluator=eval_plugin,
        train_epochs=40, #config['generic']['train_epochs'],
        train_mb_size=config['generic']['train_mb_size'],
        **config['strategy']
    )

    return cl_strategy

def train_cl_method(cl_strategy, scenario, validate=False):
    """
    Avalanche Cl training loop. For each 'experience' in scenario's train_stream:

        - Trains method on experience
        - evaluates model on train_stream and test_stream
    """
    if not validate: print('Starting experiment...')

    for experience in scenario.train_stream:
        if not validate: print(f'Start of experience: {experience.current_experience}')
        cl_strategy.train(experience, eval_streams=[scenario.train_stream, scenario.test_stream])
        if not validate: print('Training completed', '\n\n')

    if validate:
        return cl_strategy.evaluator.get_last_metrics()
    else:
        return cl_strategy.evaluator.get_all_metrics()

def training_loop(config, data, domain, outcome, model_name, strategy_name, validate=False, checkpoint_dir=None):
    """
    Training wrapper:
        - loads data
        - instantiates model
        - equips model with CL strategy
        - trains and evaluates method
        - returns either results or hyperparam optimisation if `validate`
    """

    # Loading data into 'stream' of 'experiences' (tasks)
    if not validate: print('Loading data...')
    scenario, n_tasks, n_timesteps, n_channels, weight = data_processing.load_data(data, domain, outcome, validate)
    if weight is not None:
        weight = weight.to(DEVICE)
    if not validate: print('Data loaded.\n')
    if not validate: print(f'N timesteps: {n_timesteps}\n'
                           f'N features:  {n_channels}')

    model = models.MODELS[model_name](n_channels, n_timesteps, **config['model'])
    cl_strategy = load_strategy(model, model_name, strategy_name, data, domain, n_tasks=n_tasks, weight=weight, validate=validate, config=config, benchmark=scenario)
    results = train_cl_method(cl_strategy, scenario, validate=validate)

    if validate:
        loss = results['Loss_Stream/eval_phase/test_stream/Task000']
        accuracy = results['Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000']
        balancedaccuracy = results['BalancedAccuracy_On_Trained_Experiences/eval_phase/test_stream/Task000']
        #sensitivity = results['Sens_Stream/eval_phase/test_stream/Task000']
        #specificity = results['Spec_Stream/eval_phase/test_stream/Task000']
        #precision = results['Prec_Stream/eval_phase/test_stream/Task000']
        #rocauc = results['ROCAUC_Stream/eval_phase/test_stream/Task000']
        #auprc = results['AUPRC_Stream/eval_phase/test_stream/Task000']

        # WARNING: `return` overwrites raytune report
        tune.report(loss=loss, 
                    accuracy=accuracy, 
                    balancedaccuracy=balancedaccuracy,
                    #auprc=auprc,
                    #rocauc=rocauc
        )

    else:
        return results

def hyperparam_opt(config, data, domain, outcome, model_name, strategy_name, num_samples):
    """
    Hyperparameter optimisation for the given model/strategy.
    Runs over the validation data for the first 2 tasks.
    """

    reporter = tune.CLIReporter(metric_columns=['loss', 
                                                'accuracy', 
                                                'balancedaccuracy',
                                                #'auprc',
                                                #'rocauc'
                                                ])
    resources = {'cpu':4, 'gpu':0.5} if CUDA else {'cpu':1}

    result = tune.run(
        partial(training_loop,
                data=data,
                domain=domain,
                outcome=outcome,
                model_name=model_name,
                strategy_name=strategy_name,
                validate=True),
        config=config,
        num_samples=num_samples,
        progress_reporter=reporter,
        raise_on_failed_trial=False,
        resources_per_trial=resources,
        name=f'{model_name}_{strategy_name}',
        local_dir=RESULTS_DIR / 'log' / 'raytune' / f'{data}_{outcome}_{domain}',
        trial_name_creator=lambda t: f'{model_name}_{strategy_name}_{t.trial_id}')

    best_trial = result.get_best_trial('balancedaccuracy', 'max', 'last')
    print(f'Best trial config:                             {best_trial.config}')
    print(f'Best trial final validation loss:              {best_trial.last_result["loss"]}')
    print(f'Best trial final validation accuracy:          {best_trial.last_result["accuracy"]}')
    print(f'Best trial final validation balanced accuracy: {best_trial.last_result["balancedaccuracy"]}')

    return best_trial.config

def main(data, domain, outcome, models, strategies, dropout=False, config_generic={}, config_model={}, config_cl={}, validate=False, num_samples=50):
    """
    Main training loop. Defines dataset given outcome/domain 
    and evaluates model/strategies over given hyperparams over this problem.
    """

    # Container for metrics results
    res = {m:{s:[] for s in strategies} for m in models}

    for model in models:
        for strategy in strategies:
            # Garbage collection
            torch.cuda.empty_cache()

            if validate: # Hyperparam opt over first 2 tasks
                # Load generic tuned hyper-params
                if strategy == 'Naive':
                    config = {'generic':config_generic, 'model':config_model[model], 'strategy':config_cl.get(strategy,{})}
                else:
                    naive_params = load_params(data, domain, outcome, model, 'Naive')
                    config = {'generic':naive_params['generic'], 'model':naive_params['model'], 'strategy':config_cl.get(strategy,{})}

                # JA: Investigate adding dropout to CNN (final FC layers only?)
                if not dropout and model != 'CNN':
                    config['model']['dropout'] = 0

                best_params = hyperparam_opt(config, data, domain, outcome, model, strategy, num_samples=1 if strategy=='Naive' else num_samples)
                save_params(data, domain, outcome, model, strategy, best_params)

            else: # Training loop over all tasks
                config = load_params(data, domain, outcome, model, strategy)

                # Multiple runs for Confidence Intervals
                n_repeats=5
                for _ in range(n_repeats): 
                    curr_results = training_loop(config, data, domain, outcome, model, strategy)
                    res[model][strategy].append(curr_results)

    if not validate:
        save_results(data, outcome, domain, res)
        plotting.plot_all_figs(data, domain, outcome)
