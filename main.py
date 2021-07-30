from training import main
from data_processing import plot_demos
from torch.optim import SGD, Adam

from pathlib import Path
from ray import tune

if __name__ == "__main__":
    # Specify dataset
    data='random'

    # Specify models
    models = ['MLP','CNN','RNN','LSTM']

    # Specify CL strategies
    output_dir = Path(r'C:\Users\jacob\OneDrive\Documents\code\cl code\ehr')

    # Hyperparams for grid search
    config_generic = {'lr':tune.loguniform(1e-4, 1e-1), 
                      'optimizer':tune.choice([SGD, Adam])}

    config_cl ={'Replay':{'mem_size':tune.choice([2,5,10])},
                'EWC':{'ewc_lambda':tune.loguniform(1e-3, 1e2)},
                'SI':{'si_lambda':tune.loguniform(1e-3, 1e2)},
                'LwF':{'alpha':tune.loguniform(1e-3, 1e2), 'temperature':[0.5,1,1.5,2]}}

    # Then train over all tasks
    main(data=data, output_dir=output_dir, models=models, config_generic=config_generic, config_cl=config_cl)

    # Plotting 
    if False:
        plot_demos()