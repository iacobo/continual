import json
import torch
from ray import tune
from torch import nn, optim 
from pathlib import Path
from functools import partial

from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, StreamConfusionMatrix

# Local imports
from utils import models, plotting, data_processing

# Suppressing erroneous MaxPool1d named tensors warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# GLOBALS
RESULTS_DIR = Path(__file__).parents[1] / 'results'
CONFIG_DIR = Path(__file__).parents[1] / 'config'
CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if CUDA else 'cpu'

def load_strategy(model, model_name, strategy_name, weight=None, validate=False, config={}, benchmark=None):
    """
    - `stream`     Avg accuracy over all experiences (may rely on tasks being roughly same size?)
    - `experience` Accuracy for each experience
    """

    strategy = models.STRATEGIES[strategy_name]
    criterion = nn.CrossEntropyLoss(weight=weight)

    if config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    elif config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Loggers
    # JA: subfolders for datset / experiments VVV
    interactive_logger = InteractiveLogger()
    tb_logger = TensorboardLogger(tb_log_dir = RESULTS_DIR / 'log' / 'tb_results' / f'tb_data_{plotting.get_timestamp()}' / model_name / strategy_name)

    if validate:
        loggers = [tb_logger]
    else:
        loggers = [interactive_logger, tb_logger]

    eval_plugin = EvaluationPlugin(
        StreamConfusionMatrix(save_image=False),
        accuracy_metrics(stream=True, experience=not validate),
        loss_metrics(stream=True, experience=not validate),
        loggers=loggers,
        benchmark=benchmark)

    # JA: IMPLEMENT specificity, precision etc
    # https://github.com/ContinualAI/avalanche/blob/master/notebooks/from-zero-to-hero-tutorial/05_evaluation.ipynb

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
        cl_strategy.train(experience, eval_streams=[scenario.test_stream])
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
    scenario, n_tasks, n_timesteps, n_channels, weight = data_processing.load_data(data, demo, validate)
    print('Data loaded.')
    print(f'    N tasks: {n_tasks}\n'
          f'N timesteps: {n_timesteps}\n'
          f' N features: {n_channels}')

    # JA:
    # Load main data first as .np file
    # Then call CL split on given domain increment

    model = models.MODELS[model_name](n_channels=n_channels, seq_len=n_timesteps, hidden_dim=config['hidden_dim'], **config['model'])
    cl_strategy = load_strategy(model, model_name, strategy_name, weight=weight, validate=validate, config=config, benchmark=scenario)
    results = train_cl_method(cl_strategy, scenario, validate=validate)

    if validate:
        loss = results['Loss_Stream/eval_phase/test_stream/Task000']
        accuracy = results['Top1_Acc_Stream/eval_phase/test_stream/Task000']

        # WARNING: `return` overwrites raytune report
        tune.report(loss=loss, accuracy=accuracy)

    else:
        return results

# JA: Change hyperparam directory to utils/config/hyperparams - keep search space and optimal vals in same location

def hyperparam_opt(config, data, demo, model_name, strategy_name, num_samples=10):
    """
    Hyperparameter optimisation for the given model/strategy.
    Runs over the validation data for the first 2 tasks.

    Can use returned optimal values to later run full training and testing over all n>=2 tasks.
    """

    reporter = tune.CLIReporter(metric_columns=['loss', 'accuracy'])
    resources = {'gpu': 0.25} if CUDA else {}
    
    result = tune.run(
        partial(training_loop, data=data, demo=demo, model_name=model_name, strategy_name=strategy_name, validate=True),
        config=config,
        progress_reporter=reporter,
        num_samples=num_samples,
        local_dir=RESULTS_DIR / 'log' / 'ray_results' / f'{data}_{demo}',
        name=f'{model_name}_{strategy_name}',
        trial_name_creator=lambda t: f'{model_name}_{strategy_name}_{t.trial_id}',
        resources_per_trial=resources)

    best_trial = result.get_best_trial('loss', 'min', 'last')
    print(f'Best trial config:                    {best_trial.config}')
    print(f'Best trial final validation loss:     {best_trial.last_result["loss"]}')
    print(f'Best trial final validation accuracy: {best_trial.last_result["accuracy"]}')

    return best_trial.config


# JA: Move this to main.py?
def main(data='random', demo='region', models=['MLP'], strategies=['Naive'], config_generic={}, config_model={}, config_cl={}, validate=False):

    """
    Main training loop. Takes dataset, demographic splits, 
    and evaluates model/strategies over given hyperparams over this problem.
    """

    # TRAINING 
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
            # JA: (Need to rerun multiple times for mean + CI's)
            # for i in range(5): res[model][strategy].append(...)
            else:
                config = config_cl[model][strategy]
                res[model][strategy] = training_loop(config, data, demo, model, strategy)

    if validate:
        # JA: need to save each exp/model/strat combo to a new file
        config_file = CONFIG_DIR / f'best_config_{data}_{demo}.json'
        config_file.parent.mkdir(exist_ok=True, parents=True)
        with open(config_file, 'w') as handle:
            json.dump(res, handle)
        return res
        
    # PLOTTING
    else:
        # Locally saving results
        results_file = RESULTS_DIR / 'metrics' / f'results_{data}_{demo}.json'
        results_file.parent.mkdir(exist_ok=True, parents=True)
        with open(results_file, 'w') as handle:
            res_no_tensors = {m:{s:{metric:value for metric, value in metrics.items() if 'Confusion' not in metric} for s, metrics in strats.items()} for m, strats in res.items()}
            json.dump(res_no_tensors, handle)

        plotting.plot_all_model_strats(models, strategies, data, demo, res, results_dir=RESULTS_DIR, savefig=True)

        return res