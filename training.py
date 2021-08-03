import time
import platform
from pathlib import Path
from datetime import datetime
from functools import partial
from matplotlib import pyplot as plt

# ML imports
from ray import tune
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics

# Local imports
from models import MODELS_DICT, STRATEGIES_DICT
from plotting import plot_accuracy, clean_plot
from data_processing import load_data


# HELPER FUNCTIONS

def load_strategy(model, model_name, strategy_name, train_epochs=20, eval_every=1, train_mb_size=128, eval_mb_size=1024, weight=None, output_dir=Path(''), timestamp='', validate=False, experience=False, stream=True, **config):
    """
    """
    if config['optimizer'] == 'SGD':
        optimizer = SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    elif config['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])

    criterion = CrossEntropyLoss(weight=weight)

    strategy = STRATEGIES_DICT[strategy_name]

    # Loggers
    interactive_logger = InteractiveLogger()
    tb_logger = TensorboardLogger(tb_log_dir = output_dir / f'tb_data_{timestamp}' / model_name / strategy_name) # JA ROOT

    if validate:
        loggers = [tb_logger]
    else:
        loggers = [interactive_logger, tb_logger]

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=stream,      # Avg accuracy over all experiences (may rely on tasks being roughly same size?)
                         experience=experience), # Accuracy for each experience
        loss_metrics(stream=stream, experience=experience),
        loggers=loggers)

    model = strategy(
        model, optimizer=optimizer, 
        criterion=criterion,
        train_mb_size=train_mb_size, eval_mb_size=eval_mb_size,
        train_epochs=train_epochs, eval_every=eval_every, evaluator=eval_plugin,
        **{k:v for k, v in config.items() if k not in ('optimizer','lr')} # JA: Need to make this more elegant. Take names from generic keys?
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
        return results

    else:
        results = cl_strategy.evaluator.get_all_metrics()
        return results

def training_loop(config, data, demo, model_name, strategy_name, output_dir, timestamp, validate=False):
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
    scenario, n_tasks, n_timesteps, n_channels = load_data(data, demo, output_dir, validate)
    print('Data loaded.')

    model = MODELS_DICT[model_name](n_channels=n_channels, seq_len=n_timesteps)
    cl_strategy = load_strategy(model, model_name, strategy_name, weight=None, output_dir=output_dir, timestamp=timestamp, validate=validate, **config)
    results = train_method(cl_strategy, scenario, eval_on_test=False, validate=validate)

    if validate:
        print(results.keys())
        input('...')
        if platform.system() == 'Windows':
            loss = results['Loss_Stream/eval_phase/train_stream']
            accuracy = results['Top1_Acc_Stream/eval_phase/train_stream']
        else:
            loss = results['Loss_Stream/eval_phase/train_stream/Task000']
            accuracy = results['Top1_Acc_Stream/eval_phase/train_stream/Task000']

        tune.report(loss=loss, accuracy=accuracy)
        # WARNING: `return` overwrites raytune report

    else:
        return results

def trial_str_creator(trial):
    """
    Function to give meaningful name to ray tune trial.
    """
    return f'{trial.trainable_name}_{trial.trial_id}'

def hyperparam_opt(config, data, demo, model_name, strategy_name, output_dir, timestamp):
    """
    Hyperparameter optimisation for the given model/strategy.
    Runs over the validation data for the first 2 tasks.

    Can use returned optimal values to later run full training and testing over all n>=2 tasks.
    """

    reporter = tune.CLIReporter(metric_columns=["loss", "accuracy"])
    
    result = tune.run(
        partial(training_loop, data=data, demo=demo, model_name=model_name, strategy_name=strategy_name, output_dir=output_dir, timestamp=timestamp, validate=True),
        config=config,
        progress_reporter=reporter,
        num_samples=20,
        local_dir=output_dir / 'ray_results' / f'{data}_{demo}',
        name=f'{model_name}_{strategy_name}',
        trial_name_creator=trial_str_creator,
        resources_per_trial={"cpu":4})

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f'Best trial config: {best_trial.config}')
    print(f'Best trial final validation loss: {best_trial.last_result["loss"]}')
    print(f'Best trial final validation accuracy: {best_trial.last_result["accuracy"]}')

    return best_trial.config


def main(data='random', demo='region', models=['MLP'], strategies=['Naive'], output_dir=Path('.'), config_generic=None, config_cl=None, validate=False):

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
                try: config = {**config_generic, **config_cl[strategy_name]}
                except KeyError: config = config_generic

                best_params = hyperparam_opt(config, data, demo, model_name, strategy_name, output_dir, timestamp)
                res[model_name][strategy_name] = best_params
            else:
                config = config_cl[model_name][strategy_name]
                res[model_name][strategy_name] = training_loop(config, data, demo, model_name, strategy_name, output_dir, timestamp)

            # Secondary experiment: how sensitive regularization strategies are to hyperparams
            # Tune hyperparams over increasing number of tasks?

    if validate:
        return res

    # PLOTTING
    else:
        fig, axes = plt.subplots(len(models), len(strategies), sharex=True, sharey=True, figsize=(8,8*(len(models)/len(strategies))), squeeze=False)

        for i, model in enumerate(models):
            for j, strategy in enumerate(strategies):
                plot_accuracy(strategy, model, res[model][strategy], axes[i,j])

        clean_plot(fig, axes)
        plt.savefig(output_dir / 'figs' / f'fig_{timestamp}.png')
        #plt.show()

        return res