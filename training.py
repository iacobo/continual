# ML imports
from matplotlib import pyplot as plt
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

# CL imports
from avalanche.training.strategies import Naive, JointTraining, Cumulative #Baselines
from avalanche.training.strategies import EWC, LwF, SynapticIntelligence   #Regularisation
from avalanche.training.strategies import Replay, GEM, AGEM                #Rehearsal
#from avalanche.training.strategies import AR1, CWRStar, GDumb, CoPE, StreamingLDA #???
from avalanche.benchmarks.generators import tensors_benchmark, ni_benchmark

from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger

# Local imports
from models import SimpleMLP, SimpleCNN, SimpleLSTM, SimpleRNN
from plotting import plot_accuracy, clean_plot, clean_subplot
from data_processing import generate_experiences, generate_eeg_experiences, generate_permuted_eeg_experiences

#############
# HELPER FUNCTIONS
#############

def load_strategy(model, strategy, train_epochs=10, eval_every=1, train_mb_size=128, eval_mb_size=1024, **kwargs):
    """
    """
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    strategies = {'Naive':Naive, 'Joint':JointTraining, 'Cumulative':Cumulative,
    'EWC':EWC, 'LwF':LwF, 'SI':SynapticIntelligence, 'Replay':Replay, 
    'GEM':GEM, 'AGEM':AGEM}

    strategy = strategies[strategy]
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True),# epoch=True, stream=True),
        loggers=[interactive_logger])

    model = strategy(
        model, optimizer, criterion,
        train_mb_size=train_mb_size, eval_mb_size=eval_mb_size,
        train_epochs=train_epochs, eval_every=eval_every, evaluator=eval_plugin,
        **kwargs
    )

    return model

def train_method(cl_strategy, scenario):
    """
    """
    print('Starting experiment...')

    for experience in scenario.train_stream:
        print(f'Start of experience: {experience.current_experience}')

        cl_strategy.train(experience, eval_streams=[scenario.test_stream])
        print('Training completed')

    results = cl_strategy.evaluator.get_all_metrics()

    return results

######################
# DATA LOADING
######################
# Loading data into 'stream' of 'experiences' (tasks)

GENERATE_DATA = True
USE_EEG_DATA = not GENERATE_DATA

n_tasks = 4

# Definition of training experiences
# of form (x:(patients, variables, time_steps), y:(outcome))

if GENERATE_DATA:
    n_timesteps = 6
    n_channels = 2
    experiences = generate_experiences(n_timesteps, n_channels, n_tasks)
    test_experiences = generate_experiences(n_timesteps, n_channels, n_tasks, test=True)

elif USE_EEG_DATA:
    # If you use too many task-splits, there will be all 0's or all 1's in test datasets
    experiences, test_experiences = generate_eeg_experiences(n_tasks)
    #experiences, test_experiences = generate_permuted_eeg_experiences(n_tasks)

else:
    input('No data specified... (Press Ctrl+C to quit')

n_timesteps = experiences[0][0].shape[-2]
n_channels = experiences[0][0].shape[-1]

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

print('Data loaded!')


######################
# Define models
######################

cnn = SimpleCNN(n_channels=n_channels, seq_len=n_timesteps)
mlp = SimpleMLP(n_channels=n_channels, seq_len=n_timesteps)
rnn = SimpleRNN(n_channels=n_channels, seq_len=n_timesteps)
lstm = SimpleLSTM(n_channels=n_channels, seq_len=n_timesteps)

models = {'MLP':SimpleMLP, 'CNN':SimpleCNN} #, 'RNN':SimpleRNN, 'LSTM':SimpleLSTM}

print('Models defined!')


######################
# Define CL strategy
######################

# Need to rerun multiple times, take averages, shuffle each task data internally
strategies = {'Naive':{}, 'Cumulative':{}, 'Replay':{'mem_size':10}, 'EWC':{'ewc_lambda':0.01}}

#####################
# TRAINING
#####################

# Container for metrics for plotting
res = {m:{s:None for s in strategies.keys()} for m in models.keys()}

for i, (model_name, m) in enumerate(models.items()):
    # Save initial weights for iterating over methods
    #sd = model.state_dict()
    for j, (strategy_name, args) in enumerate(strategies.items()):
        # Reload initial weights for each method
        #model.load_state_dict(sd)
        model = models[model_name](n_channels=n_channels, seq_len=n_timesteps)
        cl_strategy = load_strategy(model, strategy_name, **args)
        results = train_method(cl_strategy, scenario)
        res[model_name][strategy_name] = results


############
# PLOTTING
############

fig, axes = plt.subplots(len(models), len(strategies), sharex=True, sharey=True, figsize=(8,8*(len(models)/len(strategies))))

for i, model in enumerate(models.keys()):
    for j, strategy in enumerate(strategies.keys()):
        plot_accuracy(strategy, model, res[model][strategy], axes[i,j])
        clean_subplot(i, j, axes)

clean_plot(fig, axes)

if False:
    for _, m in models.items():
        numel = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(numel)
