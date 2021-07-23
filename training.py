import time
import datetime
from matplotlib import pyplot as plt
from pathlib import Path
import copy

# ML imports
from torch.optim import SGD #, Adam
from torch.nn import CrossEntropyLoss
#from ray import tune

# CL imports
from avalanche.training.strategies import Naive, JointTraining, Cumulative #Baselines
from avalanche.training.strategies import EWC, LwF, SynapticIntelligence   #Regularisation
from avalanche.training.strategies import Replay, GEM, AGEM                #Rehearsal
from avalanche.training.strategies import AR1, CWRStar, GDumb, CoPE, StreamingLDA #Misc

from avalanche.benchmarks.generators import tensors_benchmark
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, TensorboardLogger

# Local imports
from models import SimpleMLP, SimpleCNN, SimpleLSTM, SimpleRNN
from plotting import plot_accuracy, clean_plot
from data_processing import eicu_to_tensor


# HELPER FUNCTIONS

def load_strategy(model, model_name, strategy_name, lr=0.01, train_epochs=20, eval_every=1, train_mb_size=128, eval_mb_size=1024, weight=None, **kwargs):
    """
    """
    if model_name == 'MLP':
        lr = 0.001
    elif model_name == 'CNN':
        lr = 0.01
    elif model_name == 'LSTM':
        lr = 0.01
    else:
        lr = 0.001

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = CrossEntropyLoss(weight=weight)

    strategies = {'Naive':Naive, 'Joint':JointTraining, 'Cumulative':Cumulative,
    'EWC':EWC, 'LwF':LwF, 'SI':SynapticIntelligence, 'Replay':Replay, 
    'GEM':GEM, 'AGEM':AGEM, 'GDumb':GDumb, 'CoPE':CoPE, 
    'AR1':AR1, 'StreamingLDA':StreamingLDA, 'CWRStar':CWRStar}

    global timestamp

    strategy = strategies[strategy_name.split('_')[0]]
    interactive_logger = InteractiveLogger() #implement tensorboard
    tb_logger = TensorboardLogger(tb_log_dir=f'./tb_data_{timestamp}/{model_name}/{strategy_name}')

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True, experience=True), #Stream for avg accuracy over all over all experiences, experience=True for individuals (Former may rely on tasks being roughly same size?)
        loggers=[interactive_logger, tb_logger])

    if strategy_name == 'StreamingLDA':
        if model_name == 'CNN':
            kwargs['input_size'] = (30//4)*(512//4)
            kwargs['output_layer_name'] = 'cnn_layers'
        model = strategy(
            model, criterion=criterion,
            train_mb_size=train_mb_size, eval_mb_size=eval_mb_size,
            train_epochs=train_epochs, eval_every=eval_every, evaluator=eval_plugin,
            **kwargs
        )

    else:
        model = strategy(
            model, optimizer=optimizer, criterion=criterion,
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

def hyperparam_gridsearch(model_name, strategy_name):
    """
    Performs hyperparam gridearch on training data from first task.
    """
    # Generic hyper-params
    #config = {'lr': tune.grid_search([0.1,0.01,0.001,0.0001])}#,
              #'nl': tune.grid_search([1,2]),
              #'hs': tune.grid_search([128,256,512,1024]),
              #'optim': tune.grid_search([SGD, Adam])}

    # CL hyper-params
    # https://arxiv.org/pdf/2103.07492.pdf
    ewc_lambda = [0.1,1,10,100,1000]
    lwf_alpha = [0.1,1,10,100,1000]
    memory = [8,16,32,64,128,256]

    #analysis = tune.run()

if __name__ == "__main__":

    # Timestamp
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

    # Loading data into 'stream' of 'experiences' (tasks)
    USE_EICU_DATA = True
    USE_MIMIC_DATA = False
    weight = None

    # Definition of training experiences
    # of form (x:(patients, variables, time_steps), y:(outcome))

    if USE_EICU_DATA:
        print('Loading data...')
        experiences = eicu_to_tensor(demographic='region', balance=True)
        print('Loading test data...')
        test_experiences = copy.deepcopy(experiences)
        print('Data loaded.')

        #if not balance: weight = torch.tensor([1.4, 0.2])

    else:
        input('No data specified... (Press Ctrl+C to quit')

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
    models = {'MLP':SimpleMLP, 'CNN':SimpleCNN, 'RNN':SimpleRNN, 'LSTM':SimpleLSTM}
    models = {'CNN':SimpleCNN, 'MLP':SimpleMLP}

    # Define CL strategy
    #strategies = {'Naive':{}, 'Cumulative':{}, 'Replay':{'mem_size':20}, 'EWC':{'ewc_lambda':100}}
    strategies = {'Naive':{}, 'SI_1e-1':{'si_lambda':0.1}, 'SI_1':{'si_lambda':1}, 'SI_10':{'si_lambda':10}, 'SI_100':{'si_lambda':100}} 
    #strategies = {'LwF_1':{'alpha':1, 'temperature':0.5}, 'LwF_1000':{'alpha':1, 'temperature':2}}
    strategies = {'Naive':{}, 'Cumulative':{}, 'Replay':{'mem_size':50}, 'EWC':{'ewc_lambda':100}, 'StreamingLDA':{'input_size':int(0.4*512), 'num_classes':2, 'output_layer_name':'features'}}

    # TRAINING (Need to rerun multiple times, take averages)
    # Container for metrics for plotting
    res = {m:{s:None for s in strategies.keys()} for m in models.keys()}

    for model_name in models.keys():
        for strategy_name, args in strategies.items():
            model = models[model_name](n_channels=n_channels, seq_len=n_timesteps)
            cl_strategy = load_strategy(model, model_name, strategy_name, weight=weight, **args)
            results = train_method(cl_strategy, scenario, eval_on_test=False)
            res[model_name][strategy_name] = results


    # PLOTTING
    fig, axes = plt.subplots(len(models), len(strategies), sharex=True, sharey=True, figsize=(8,8*(len(models)/len(strategies))), squeeze=False)

    for i, model in enumerate(models.keys()):
        for j, strategy in enumerate(strategies.keys()):
            plot_accuracy(strategy, model, res[model][strategy], axes[i,j])

    clean_plot(fig, axes)

    ROOT = Path(r'C:\Users\jacob\OneDrive\Documents\code\cl code\ehr\figs')
    plt.savefig(ROOT / f'fig_{timestamp}.png')

    if False:
        for _, m in models.items():
            numel = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(numel)