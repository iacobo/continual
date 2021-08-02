import training
from data_processing import plot_demos

from pathlib import Path
from ray import tune

def main(validate=False):
    # Specify dataset
    data = 'random'
    output_dir = Path(r'C:\Users\jacob\OneDrive\Documents\code\cl code\ehr')

    # Specify models
    models = ['MLP'] #,'CNN','RNN','LSTM']

    # Specify CL strategies
    strategies = ['Naive', 'EWC'] #'Cumulative', 'Replay', 'SI', 'LwF', 'MAS', 'GEM', 'AGEM'

    # Generic hyperparameter search-space
    config_generic = {'lr':tune.loguniform(1e-4, 1e-1), 
                      'optimizer':tune.choice(['SGD','Adam'])}
                      # 'batch_size':tune.choice([16,32,64,128,256,512,1024])
                      # 'hl':tune.choice([64,128,256,512,1024])
                      # 'nl':tune.choice(['tan', 'relu'])

    # CL hyper-params
    # https://arxiv.org/pdf/2103.07492.pdf
    config_cl ={'Replay':{'mem_size':tune.choice([2,5,10])},
                'EWC':{'ewc_lambda':tune.loguniform(1e-3, 1e2)},
                'SI':{'si_lambda':tune.loguniform(1e-3, 1e2)},
                'LwF':{'alpha':tune.loguniform(1e-3, 1e2), 'temperature':tune.uniform(0.0,3.0)}}

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