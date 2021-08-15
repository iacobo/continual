import argparse
from ray import tune
from utils import training, plotting
import numpy as np

def get_dropout_from_n_layers(spec):
    """
    Has dropout of 0 if n_layers==1, else random dropout.
    """
    if spec.config.model.n_layers == 1:
        return 0.0
    else:
        p = np.random.randint(0,6)
        return p * 0.1

def main(args):

    # Specify models
    if args.models=='all':
        args.models = ['MLP', 'CNN', 'RNN', 'LSTM']

    # Specify CL strategies
    if args.strategies=='all':
        args.strategies = ['Naive', 'Cumulative', 'EWC', 'SI', 'LwF', 'Replay', 'GDumb', 'GEM', 'AGEM'] # JA: INVESTIGATE MAS!!!

    # Generic hyperparameter search-space
    config_generic = {'lr':tune.choice([1e-4,1e-3,1e-2,1e-1]), 
                      'optimizer':tune.choice(['SGD','Adam']),
                      'hidden_dim':tune.choice([64,128,256,512,1024]),
                      'train_epochs':50, 
                      'train_mb_size':tune.choice([32,64,128,256,512,1024])
                      }

    config_model = {# JA: use ModuleList(?) to parameterise n_layers for MLP and CNN
                    'CNN':{'nonlinearity':tune.choice(['tanh', 'relu'])},
                    'MLP':{'dropout':tune.choice([0,0.1,0.2,0.3,0.4,0.5]), 'nonlinearity':tune.choice(['tanh', 'relu'])},
                    'RNN':{'n_layers':tune.choice([1,2,3,4]), 'dropout':tune.sample_from(get_dropout_from_n_layers), 'bidirectional':tune.choice([True,False]), 'nonlinearity':tune.choice(['tanh', 'relu'])},
                    'LSTM':{'n_layers':tune.choice([1,2,3,4]), 'dropout':tune.sample_from(get_dropout_from_n_layers), 'bidirectional':tune.choice([True,False])}
                    }

    config_cl ={'Replay':{'mem_size':tune.choice([4,16,32])},
                'GDumb':{'mem_size':tune.choice([4,16,32])},
                'EWC':{'ewc_lambda':tune.choice([1e-3,1e-2,1e-1,1e0,1e1,1e2]), 'mode':tune.choice(['separate','onlinesum'])},
                'SI':{'si_lambda':tune.choice([1e-3,1e-2,1e-1,1e0,1e1,1e2])},
                'LwF':{'alpha':tune.choice([1e-3,1e-2,1e-1,1e0,1e1,1e2]), 'temperature':tune.quniform(0,3,0.5)},
                'GEM':{'patterns_per_exp':tune.choice([4,16,32]), 'memory_strength':tune.quniform(0,1,0.1)},
                'AGEM':{'patterns_per_exp':tune.choice([4,16,32]), 'sample_size':tune.choice([4,16,32])}
                #'CoPE':
                }

    # Hyperparam opt over validation data for first 2 tasks
    if args.validate:
        best_params = training.main(data=args.data, demo=args.experiment, models=args.models, strategies=args.strategies, config_generic=config_generic, config_model=config_model, config_cl=config_cl, validate=True)
        # Save to local file
    else:
        best_params = None
    
    # Train and test over all tasks
    # JA: MUST ENSURE DATA SPLIT IS SAME FOR HYPERPARAM TUNE AND FURTHER TRAINING!!!!
    training.main(data=args.data, demo=args.experiment, models=args.models, strategies=args.strategies, config_generic={}, config_cl=best_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', 
                        type=str, 
                        default='fiddle_mimic3', 
                        choices=['fiddle_mimic3','fiddle_eicu','random'], 
                        help='Dataset to use.')
    parser.add_argument('--outcome', 
                        type=str, 
                        default='mortality_48h', 
                        choices=['ARF_4h','ARF_12h','shock_4h','shock_12h','mortality_48h'], 
                        help='Outcome to predict.')
    parser.add_argument('--experiment', 
                        type=str, 
                        default='age', 
                        choices=['time_month','time_season','time_year','region','hospital','age','sex','ethnicity'], 
                        help='Experiment to run.') # Domain incremental
    parser.add_argument('--strategies', 
                        type=str, 
                        default='all', 
                        choices=['Naive', 'Cumulative', 'EWC', 'SI', 'LwF', 'Replay', 'GDumb', 'GEM', 'AGEM'], 
                        nargs='+',
                        help='Continual learning strategy(s) to evaluate.')
    parser.add_argument('--models', 
                        type=str, 
                        default='all', 
                        choices=['MLP','CNN','RNN','LSTM'], 
                        nargs='+',
                        help='Model(s) to evaluate.')
    parser.add_argument('--validate', 
                        action='store_const', 
                        const=True, 
                        default=False, 
                        help='Tune hyperparameters.')
    args = parser.parse_args()

    main(args)

    # Secondary experiments:
    ########################
    # Sensitivity to sequence length (4hr vs 12hr)
    # Sensitivity to replay size Naive -> replay -> Cumulative
    # Sensitivity to hyperparams of reg methods
    # Sensitivity to number of variables
    #  JA: Secondary experiment: how sensitive regularization strategies are to hyperparams
    #  Tune hyperparams over increasing number of tasks?

    # Plotting
    if False:
        plotting.plot_demos()
