import time
import json
from pathlib import Path
from datetime import datetime
from functools import partial
from matplotlib import pyplot as plt

# ML imports
import torch
from ray import tune
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, StreamConfusionMatrix

# Local imports
from utils import models, plotting, data_processing

RESULTS_DIR = Path(__file__).parents[1] / 'results'


# HELPER FUNCTIONS

def get_timestamp():
    """
    Returns current timestamp as string.
    """
    ts = time.time()
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

def load_strategy(model, model_name, strategy_name, weight=None, validate=False, config={}):
    """
    - `stream`     Avg accuracy over all experiences (may rely on tasks being roughly same size?)
    - `experience` Accuracy for each experience
    """

    strategy = models.STRATEGIES[strategy_name]
    criterion = CrossEntropyLoss(weight=weight)

    if config['optimizer'] == 'SGD':
        optimizer = SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    elif config['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])

    # Loggers
    # JA: subfolders for datset / experiments VVV
    interactive_logger = InteractiveLogger()
    tb_logger = TensorboardLogger(tb_log_dir = RESULTS_DIR / 'tb_results' / f'tb_data_{get_timestamp()}' / model_name / strategy_name)

    if validate:
        loggers = [tb_logger]
    else:
        loggers = [interactive_logger, tb_logger]

    eval_plugin = EvaluationPlugin(
        StreamConfusionMatrix(save_image=False),
        accuracy_metrics(stream=True, experience=not validate),
        loss_metrics(stream=True, experience=not validate),
        loggers=loggers)

    # JA: IMPLEMENT specificity, precision etc
    # https://github.com/ContinualAI/avalanche/blob/master/notebooks/from-zero-to-hero-tutorial/05_evaluation.ipynb

    cl_strategy = strategy(
        model, 
        optimizer=optimizer, 
        criterion=criterion, 
        eval_mb_size=1024, 
        eval_every=1,
        evaluator=eval_plugin,
        train_epochs=config['train_epochs'],
        train_mb_size=config['train_mb_size'],
        **config['strategy']
    )

    return cl_strategy

def train_cl_method(cl_strategy, scenario, eval_on_test=True, validate=False):
    """
    Avalanche Cl training loop. For each 'experience' in scenario's train_stream:

        - Trains method on experience
        - evaluates model on test_stream
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

    if validate:
        results = cl_strategy.eval(scenario.test_stream)

    else:
        results = cl_strategy.evaluator.get_all_metrics()
    
    return results

def training_loop(config, data, demo, model_name, strategy_name, validate=False):
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
    scenario, n_tasks, n_timesteps, n_channels, weight = data_processing.load_data(data, demo, validate)
    print('Data loaded.')
    print(f'    N tasks: {n_tasks} \nN timesteps: {n_timesteps} \n N features: {n_channels}')

    # JA:
    # Load main data first as .np file
    # Then call CL split on given domain increment

    model = models.MODELS[model_name](n_channels=n_channels, seq_len=n_timesteps, hidden_dim=config['hidden_dim'], **config['model'])
    cl_strategy = load_strategy(model, model_name, strategy_name, weight=weight, validate=validate, config=config)
    results = train_cl_method(cl_strategy, scenario, eval_on_test=True, validate=validate)

    if validate:
        loss = results['Loss_Stream/eval_phase/test_stream/Task000']
        accuracy = results['Top1_Acc_Stream/eval_phase/test_stream/Task000']

        # WARNING: `return` overwrites raytune report
        tune.report(loss=loss, accuracy=accuracy)

    else:
        return results

def hyperparam_opt(config, data, demo, model_name, strategy_name):
    """
    Hyperparameter optimisation for the given model/strategy.
    Runs over the validation data for the first 2 tasks.

    Can use returned optimal values to later run full training and testing over all n>=2 tasks.
    """

    reporter = tune.CLIReporter(metric_columns=["loss", "accuracy"])
    resources = {"gpu": 0.25} if torch.cuda.is_available() else {}
    
    result = tune.run(
        partial(training_loop, data=data, demo=demo, model_name=model_name, strategy_name=strategy_name, validate=True),
        config=config,
        progress_reporter=reporter,
        num_samples=5,
        local_dir=RESULTS_DIR / 'ray_results' / f'{data}_{demo}',
        name=f'{model_name}_{strategy_name}',
        trial_name_creator=lambda t: f'{model_name}_{strategy_name}_{t.trial_id}',
        resources_per_trial=resources)

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f'Best trial config:                    {best_trial.config}')
    print(f'Best trial final validation loss:     {best_trial.last_result["loss"]}')
    print(f'Best trial final validation accuracy: {best_trial.last_result["accuracy"]}')

    return best_trial.config


def main(data='random', demo='region', models=['MLP'], strategies=['Naive'], config_generic=None, config_model=None, config_cl=None, validate=False):

    """
    data: ['random','mimic3','eicu','iord']
    demo: ['region','sex','age','ethnicity','ethnicity_coarse','hospital']
    """

    # TRAINING 
    # JA: (Need to rerun multiple times, take averages)
    # Container for metrics for plotting
    res = {m:{s:None for s in strategies} for m in models}

    for model in models:
        for strategy in strategies:
            # Hyperparam opt
            if validate:
                config = {**config_generic, 'model':config_model[model], 'strategy':config_cl.get(strategy,{})}
                best_params = hyperparam_opt(config, data, demo, model, strategy)
                res[model][strategy] = best_params
            # Training loop
            else:
                config = config_cl[model][strategy]
                res[model][strategy] = training_loop(config, data, demo, model, strategy)

            # JA: Secondary experiment: how sensitive regularization strategies are to hyperparams
            # Tune hyperparams over increasing number of tasks?

    if validate:
        return res
        
    # PLOTTING
    else:
        # Locally saving results
        with open(RESULTS_DIR / f'latest_results_{data}_{demo}.json', 'w') as handle:
            res_acc = {k:v for k,v in res.items() if 'Top1_Acc_Exp' in k}
            json.dump(res_acc, handle)

        fig, axes = plt.subplots(len(models), len(strategies), sharex=True, sharey=True, figsize=(8,8*(len(models)/len(strategies))), squeeze=False)

        for i, model in enumerate(models):
            for j, strategy in enumerate(strategies):
                plotting.plot_accuracy(strategy, model, res[model][strategy], axes[i,j])

        plotting.clean_plot(fig, axes)
        plt.savefig(RESULTS_DIR / 'figs' / f'fig_{data}_{demo}_{get_timestamp()}.png')

        return res