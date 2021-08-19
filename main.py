import json
import argparse
from pathlib import Path
from utils import training
from config import config

CONFIG_DIR = Path(__file__).parents[0] / 'config'

def main(args):
    if args.models=='all':
        args.models = ['MLP', 'CNN', 'RNN', 'LSTM']

    if args.strategies=='all':
        args.strategies = ['Naive', 'Cumulative', 'EWC', 'SI', 'LwF', 'Replay', 'GDumb', 'GEM'] # JA: INVESTIGATE MAS, 'AGEM' (num samples > mem?)!!!

    # Hyperparam optimisation over validation data for first 2 tasks
    if args.validate:
        best_params = training.main(data=args.data, 
                                    demo=args.experiment, 
                                    models=args.models, 
                                    strategies=args.strategies, 
                                    config_generic=config.config_generic, 
                                    config_model=config.config_model, 
                                    config_cl=config.config_cl, 
                                    validate=True)

    # Load previously tuned hyper-param config
    else:
        with open(CONFIG_DIR / f'config_{args.data}_{args.experiment}.json') as json_file:
            best_params = json.load(json_file)
    
    # Train and test over all tasks (using optimised hyperparams)
    training.main(data=args.data, 
                  demo=args.experiment, 
                  models=args.models, 
                  strategies=args.strategies,
                  config_cl=best_params)

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

#plotting.plot_demos()

# Secondary experiments:
########################
# Sensitivity to sequence length (4hr vs 12hr)
# Sensitivity to replay size Naive -> replay -> Cumulative
# Sensitivity to hyperparams of reg methods (Tune hyperparams over increasing number of tasks?)
# Sensitivity to number of variables