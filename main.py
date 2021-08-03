import training
from data_processing import plot_demos

import platform
from ray import tune
from pathlib import Path

def main(validate=False):

    if platform.system() == 'Linux':
        output_dir = Path('/home/scat5356/Downloads')
    elif platform.system() == 'Windows':
        output_dir = Path(r'C:\Users\jacob\OneDrive\Documents\code\cl code\ehr')

    # Specify dataset
    data = 'random'

    # Specify models
    models = ['MLP','CNN', 'RNN','LSTM']

    # Specify CL strategies
    strategies = ['Naive', 'Cumulative', 'EWC', 'SI', 'LwF', 'Replay', 'GEM'] #'AGEM' # JA: INVESTIGATE MAS!!!

    # Generic hyperparameter search-space
    config_generic = {'lr':tune.loguniform(1e-4, 1e-1), 
                      'optimizer':tune.choice(['SGD','Adam'])}
                      # 'train_mb_size':tune.choice([32,128,256,512,1024])
                      # 'hl':tune.choice([64,128,256,512,1024])
                      # 'nl':tune.choice(['tan', 'relu'])
                      # 'bilinear':tune.choice([True,False])

    # CL hyper-params
    # https://arxiv.org/pdf/2103.07492.pdf
    config_cl ={'Replay':{'mem_size':tune.choice([2,5,10])},
                'EWC':{'ewc_lambda':tune.loguniform(1e-3, 1e2)},
                'SI':{'si_lambda':tune.loguniform(1e-3, 1e2)},
                'LwF':{'alpha':tune.loguniform(1e-3, 1e2), 'temperature':tune.uniform(0.0,3.0)},
                'GEM':{'patterns_per_exp':tune.choice([2,5,10]), 'memory_strength':tune.uniform(0.0,1.0)} #32,64,128,256
                }

    # Hyperparam opt over validation data for first 2 tasks
    if validate:
        best_params = training.main(data=data, output_dir=output_dir, models=models, strategies=strategies, config_generic=config_generic, config_cl=config_cl, validate=True)
        # Save to local file
    else:
        best_params = None
    
    # Train and test over all tasks
    # JA: MUST ENSURE DATA SPLIT IS SAME FOR HYPERPARAM TUNE AND FURTHER TRAINING!!!!
    training.main(data=data, output_dir=output_dir, models=models, strategies=strategies, config_generic={}, config_cl=best_params)

    # Plotting
    if False:
        plot_demos()

if __name__ == "__main__":
    main(validate=True)

    # Secondary experiments:
    ########################
    # Sensitivity to sequence length
    # Sensitivity to replay size Naive -> Cumulative
    # Sensitivity to hyperparams of reg methods