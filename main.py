import argparse
from ray import tune
from utils import training, plotting

def main(args):

    # Specify dataset
    if args.data=='all':
        args.data = 'fiddle_mimic' #'eICU'

    if args.experiment:
        args.experiment = 'age'

    # Specify models
    if args.models=='all':
        # JA: Fix single argument passed, need to list-ify
        args.models = ['MLP', 'CNN', 'RNN', 'LSTM']

    # Specify CL strategies
    if args.strategies=='all':
        args.strategies = ['Naive', 'Cumulative', 'EWC', 'SI', 'LwF', 'Replay', 'GEM'] #'AGEM' # JA: INVESTIGATE MAS!!!

    # Generic hyperparameter search-space
    config_generic = {'lr':tune.choice([1e-4,1e-3,1e-2,1e-1]), 
                      'optimizer':tune.choice(['SGD','Adam']),
                      'hidden_dim':tune.choice([64,128,256,512,1024]),
                      'train_epochs':tune.choice([100]), 
                      'train_mb_size':tune.choice([32,64,128,256,512,1024])
                      }

    config_model = {'MLP':{'dropout':tune.choice([0,0.1,0.2,0.3,0.4,0.5]), 'nonlinearity':tune.choice(['tanh', 'relu'])},
                    'CNN':{'nonlinearity':tune.choice(['tanh', 'relu'])},
                    'RNN':{'dropout':tune.choice([0,0.1,0.2,0.3,0.4,0.5]), 'bilinear':tune.choice([True,False]), 'n_layers':tune.choice([1,2,3,4]), 'nonlinearity':tune.choice(['tanh', 'relu'])},
                    'LSTM':{'dropout':tune.choice([0,0.1,0.2,0.3,0.4,0.5]), 'bilinear':tune.choice([True,False]), 'n_layers':tune.choice([1,2,3,4])}
                    }

    # CL hyper-params
    # https://arxiv.org/pdf/2103.07492.pdf
    config_cl ={'Replay':{'mem_size':tune.choice([2,5,10])},
                'EWC':{'ewc_lambda':tune.choice([1e-3,1e-2,1e-1,1e0,1e1,1e2])},
                'SI':{'si_lambda':tune.choice([1e-3,1e-2,1e-1,1e0,1e1,1e2])},
                'LwF':{'alpha':tune.choice([1e-3,1e-2,1e-1,1e0,1e1,1e2]), 'temperature':tune.uniform(0.0,3.0)},
                'GEM':{'patterns_per_exp':tune.choice([2,5,10]), 'memory_strength':tune.uniform(0.0,1.0)} #32,64,128,256
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
                        default='all', 
                        choices=['fiddle_mimic','fiddle_eicu','MIMIC','eICU','iord','random'], 
                        help='Dataset to use.')
    parser.add_argument('--outcome', 
                        type=str, 
                        default='all', 
                        choices=['arf','shock','mortality'], 
                        help='Outcome to predict.')
    parser.add_argument('--experiment', 
                        type=str, 
                        default='all', 
                        choices=['time_month','time_season','time_year','region','hospital','age','sex','ethnicity'], 
                        help='Experiment to run.') # Domain incremental
    parser.add_argument('--strategies', 
                        type=str, 
                        default='all', 
                        choices=['Naive', 'Cumulative', 'EWC', 'SI', 'LwF', 'Replay', 'GEM'], 
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

    # Plotting
    if False:
        plotting.plot_demos()
