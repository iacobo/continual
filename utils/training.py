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

def load_strategy(model, model_name, strategy_name, eval_every=1, eval_mb_size=1024, weight=None, timestamp='', validate=False, experience=False, stream=True, config={}):
    """
    - `stream`     Avg accuracy over all experiences (may rely on tasks being roughly same size?)
    - `experience` Accuracy for each experience
    """
    if config['optimizer'] == 'SGD':
        optimizer = SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    elif config['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])

    criterion = CrossEntropyLoss(weight=weight)

    strategy = models.STRATEGIES[strategy_name]

    # Loggers
    interactive_logger = InteractiveLogger()
    # JA: subfolders for datset / experiments VVV
    tb_logger = TensorboardLogger(tb_log_dir = RESULTS_DIR / 'tb_results' / f'tb_data_{timestamp}' / model_name / strategy_name)

    if validate:
        loggers = [tb_logger]

        eval_plugin = EvaluationPlugin(
            accuracy_metrics(stream=stream, experience=experience),
            loss_metrics(stream=stream, experience=experience),
            loggers=loggers)

    else:
        loggers = [interactive_logger, tb_logger]
        experience=True

        eval_plugin = EvaluationPlugin(
            accuracy_metrics(stream=stream, experience=experience),
            loss_metrics(stream=stream, experience=experience),
            StreamConfusionMatrix(save_image=False), 
            loggers=loggers)

    model = strategy(
        model, 
        optimizer=optimizer, 
        criterion=criterion, 
        eval_mb_size=eval_mb_size, 
        eval_every=eval_every,
        evaluator=eval_plugin,
        train_epochs=config['train_epochs'],
        train_mb_size=config['train_mb_size'],
        **config['strategy']
    )

    return model

def train_method(cl_strategy, scenario, eval_on_test=True, validate=False):
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

def training_loop(config, data, demo, model_name, strategy_name, timestamp, validate=False):
    """
    Training wrapper:
        - loads data
        - instantiates model
        - equips model with CL strategy
        - trains and evaluates method
        - returns either resutls or hyperparam optimisation if `validate`

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
    cl_strategy = load_strategy(model, model_name, strategy_name, weight=weight, timestamp=timestamp, validate=validate, config=config)
    results = train_method(cl_strategy, scenario, eval_on_test=True, validate=validate)

    if validate:
        # JA: Avalanche differing behaviour in latest version?
        # JA: Need to check ability to record eval on train *and* test streams
        try:
            loss = results['Loss_Stream/eval_phase/test_stream']
            accuracy = results['Top1_Acc_Stream/eval_phase/test_stream']
        except:
            loss = results['Loss_Stream/eval_phase/test_stream/Task000']
            accuracy = results['Top1_Acc_Stream/eval_phase/test_stream/Task000']

        tune.report(loss=loss, accuracy=accuracy)
        # WARNING: `return` overwrites raytune report

    else:
        return results

def hyperparam_opt(config, data, demo, model_name, strategy_name, timestamp):
    """
    Hyperparameter optimisation for the given model/strategy.
    Runs over the validation data for the first 2 tasks.

    Can use returned optimal values to later run full training and testing over all n>=2 tasks.
    """

    reporter = tune.CLIReporter(metric_columns=["loss", "accuracy"])
    resources = {"gpu": 1} if torch.cuda.is_available() else {}
    
    result = tune.run(
        partial(training_loop, data=data, demo=demo, model_name=model_name, strategy_name=strategy_name, timestamp=timestamp, validate=True),
        config=config,
        progress_reporter=reporter,
        num_samples=20,
        local_dir=RESULTS_DIR / 'ray_results' / f'{data}_{demo}',
        name=f'{model_name}_{strategy_name}',
        trial_name_creator=lambda t: f'{model_name}_{strategy_name}_{t.trial_id}',
        resources_per_trial=resources)

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f'Best trial config: {best_trial.config}')
    print(f'Best trial final validation loss: {best_trial.last_result["loss"]}')
    print(f'Best trial final validation accuracy: {best_trial.last_result["accuracy"]}')

    return best_trial.config


def main(data='random', demo='region', models=['MLP'], strategies=['Naive'], config_generic=None, config_model=None, config_cl=None, validate=False):

    """
    data: ['random','MIMIC','eICU','iORD']
    demo: ['region','sex','age','ethnicity','ethnicity_coarse','hospital']
    """

    # Timestamp for logging
    ts = time.time()
    timestamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

    # TRAINING (Need to rerun multiple times, take averages)
    # Container for metrics for plotting CHANGE TO TXT FILE
    res = {m:{s:None for s in strategies} for m in models}

    # Change to remove ref to keys, use names directly and key, val superset
    for model_name in models:
        for strategy_name in strategies:
            # Training loop
            if validate:
                # Union generic and CL strategy-specific hyperparams
                try: config = {**config_generic, 'model':config_model[model_name], 'strategy':config_cl[strategy_name]}
                except KeyError: config = {**config_generic, 'model':config_model[model_name], 'strategy':{}}

                best_params = hyperparam_opt(config, data, demo, model_name, strategy_name, timestamp)
                res[model_name][strategy_name] = best_params
            else:
                config = config_cl[model_name][strategy_name]
                res[model_name][strategy_name] = training_loop(config, data, demo, model_name, strategy_name, timestamp)

            # Secondary experiment: how sensitive regularization strategies are to hyperparams
            # Tune hyperparams over increasing number of tasks?

    if validate:
        return res
        
    # PLOTTING
    else:
        # Locally saving results
        with open(RESULTS_DIR / f'latest_results_{data}_{demo}.json', 'w') as handle:
            # JA: cannot save tensor as json
            res_noconf = {k:v for k,v in res.items() if 'Top1_Acc_Exp' in k}
            json.dump(res_noconf, handle)

        fig, axes = plt.subplots(len(models), len(strategies), sharex=True, sharey=True, figsize=(8,8*(len(models)/len(strategies))), squeeze=False)

        for i, model in enumerate(models):
            for j, strategy in enumerate(strategies):
                plotting.plot_accuracy(strategy, model, res[model][strategy], axes[i,j])

        plotting.clean_plot(fig, axes)
        plt.savefig(RESULTS_DIR / 'figs' / f'fig_{timestamp}.png')
        #plt.show()

        return res