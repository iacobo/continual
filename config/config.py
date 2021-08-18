from ray import tune
import numpy as np

# Conditional hyper-param functions
def get_dropout_from_n_layers(spec):
    """
    Has dropout of 0 if n_layers==1, else random dropout.
    """
    if spec.config.model.n_layers==1:
        return 0
    else:
        return 0.1 * np.random.randint(0,6)

def get_decay_from_ewc_mode(spec):
    """
    Return decay factor if usig Online EWC.
    """
    if spec.config.strategy.mode=='online':
        return 0.1 * np.random.randint(0,11)
    else:
        return None

# Hyperparameter search-space
config_generic = {'lr':tune.choice([1e-4,1e-3,1e-2]), 
                  'optimizer':tune.choice(['Adam']), #'SGD',
                  'hidden_dim':tune.choice([128,256,512,1024]),
                  'train_epochs':100, 
                  'train_mb_size':tune.choice([64,128,256,512,1024])
                  }

# JA: use ModuleList(?) to parameterise n_layers for MLP and CNN
config_model = {'CNN':{'nonlinearity':tune.choice(['tanh', 'relu'])},
                'MLP':{'dropout':tune.choice([0,0.25,0.5]), 
                       'nonlinearity':tune.choice(['tanh', 'relu'])},
                'RNN':{'n_layers':tune.choice([1,2]), 
                       'dropout':tune.sample_from(get_dropout_from_n_layers), 
                       'bidirectional':tune.choice([True,False]), 
                       'nonlinearity':tune.choice(['tanh', 'relu'])},
                'LSTM':{'n_layers':tune.choice([1,2]), 
                        'dropout':tune.sample_from(get_dropout_from_n_layers), 
                        'bidirectional':tune.choice([True,False])}
                }

config_cl ={'Replay':{'mem_size':tune.choice([4,16,32])},
            'GDumb':{'mem_size':tune.choice([4,16,32])},
            'EWC':{'ewc_lambda':tune.choice([1e-3,1e-2,1e-1,1e0,1e1,1e2]), 
                   'mode':tune.choice(['separate','online']), 
                   'decay_factor':tune.sample_from(get_decay_from_ewc_mode)},
            'SI':{'si_lambda':tune.choice([1e-3,1e-2,1e-1,1e0,1e1,1e2])},
            'LwF':{'alpha':tune.choice([1e-3,1e-2,1e-1,1e0,1e1,1e2]), 
                   'temperature':tune.quniform(0,3,0.5)},
            'GEM':{'patterns_per_exp':tune.choice([4,16,32]), 
                   'memory_strength':tune.quniform(0,1,0.1)},
            'AGEM':{'patterns_per_exp':tune.choice([4,16,32]), 
                    'sample_size':tune.choice([4,16,32])}
            #'CoPE':
            }