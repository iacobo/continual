"""
Main training script.
"""

import argparse

from utils import training
from config import config

def main(args):
    """
    Runs appropriate experiment(s) from passed args.
    """
    if args.models=='all':
        args.models = ['MLP','CNN','LSTM','Transformer']

    # JA: INVESTIGATE MAS, 'AGEM' (num samples > mem?)!!!
    if args.strategies=='all':
        args.strategies = ['Naive','Cumulative','EWC','OnlineEWC','SI','LwF','Replay','GEM','AGEM']

    # Hyperparam optimisation over validation data for first 2 tasks
    if args.validate:
        training.main(data=args.data,
                      domain=args.domain_shift,
                      outcome=args.outcome,
                      models=args.models,
                      strategies=args.strategies, 
                      dropout=args.dropout,
                      config_generic=config.config_generic,
                      config_model=config.config_model,
                      config_cl=config.config_cl,
                      num_samples=args.num_samples,
                      validate=True)

    # Train and test over all tasks (using optimised hyperparams)
    training.main(data=args.data,
                  domain=args.domain_shift,
                  outcome=args.outcome,
                  models=args.models,
                  strategies=args.strategies)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        type=str,
                        default='mimic3',
                        choices=['mimic3','eicu','random'],
                        help='Dataset to use.')

    parser.add_argument('--outcome',
                        type=str,
                        default='mortality_48h',
                        choices=['ARF_4h','ARF_12h','Shock_4h','Shock_12h','mortality_48h'],
                        help='Outcome to predict.')

    parser.add_argument('--domain_shift',
                        type=str,
                        default='age',
                        choices=['time_season','region','hospital','age','sex','ethnicity','ethnicity_coarse'],
                        help='Domain shift exhibited in tasks.')

    parser.add_argument('--strategies',
                        type=str,
                        default='all',
                        choices=['Naive','Cumulative','EWC','OnlineEWC','SI','LwF','Replay','GDumb','GEM','AGEM'],
                        nargs='+',
                        help='Continual learning strategy(s) to evaluate.')

    parser.add_argument('--models',
                        type=str,
                        default='all',
                        choices=['MLP','CNN','RNN','LSTM','GRU','Transformer'],
                        nargs='+',
                        help='Model(s) to evaluate.')

    parser.add_argument('--dropout',
                        action='store_const',
                        const=True,
                        default=False,
                        help='Add dropout to model(s).')

    parser.add_argument('--validate',
                        action='store_const',
                        const=True,
                        default=False,
                        help='Tune hyperparameters.')

    parser.add_argument('--num_samples',
                        type=int,
                        default=50,
                        help='Number of samples to draw during hyperparameter search.')

    args = parser.parse_args()
    main(args)
