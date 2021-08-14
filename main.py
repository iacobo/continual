import argparse
from ray import tune
from utils import training, plotting

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
                      'train_epochs':tune.choice([200]), 
                      'train_mb_size':tune.choice([32,64,128,256,512,1024])
                      }

    config_model = {'CNN':{'nonlinearity':tune.choice(['tanh', 'relu'])},
                    'MLP':{'dropout':tune.choice([0,0.1,0.2,0.3,0.4,0.5]), 'nonlinearity':tune.choice(['tanh', 'relu'])},
                    'RNN':{'dropout':tune.choice([0,0.1,0.2,0.3,0.4,0.5]), 'bidirectional':tune.choice([True,False]), 'n_layers':tune.choice([1,2,3,4]), 'nonlinearity':tune.choice(['tanh', 'relu'])},
                    'LSTM':{'dropout':tune.choice([0,0.1,0.2,0.3,0.4,0.5]), 'bidirectional':tune.choice([True,False]), 'n_layers':tune.choice([1,2,3,4])}
                    }

    config_cl ={'Replay':{'mem_size':tune.choice([4,16,32])},
                'GDumb':{'mem_size':tune.choice([4,16,32])},
                'EWC':{'ewc_lambda':tune.choice([1e-3,1e-2,1e-1,1e0,1e1,1e2]), 'mode':tune.choice(['separate','onlinesum'])},
                'SI':{'si_lambda':tune.choice([1e-3,1e-2,1e-1,1e0,1e1,1e2])},
                'LwF':{'alpha':tune.choice([1e-3,1e-2,1e-1,1e0,1e1,1e2]), 'temperature':tune.quniform([0,3,0.5])},
                'GEM':{'patterns_per_exp':tune.choice([4,16,32]), 'memory_strength':tune.uniform(0.0,1.0)},
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
                        choices=['fiddle_mimic3','fiddle_eicu','MIMIC','eICU','iord','random'], 
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
                        choices=['Naive', 'Cumulative', 'EWC', 'SI', 'LwF', 'Replay', 'GEM', 'AGEM'], 
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
