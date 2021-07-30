from training import main
from data_processing import plot_demos
from torch.optim import SGD, Adam

from pathlib import Path

if __name__ == "__main__":
    # Specify dataset
    data='random'

    # Specify models
    models = ['MLP','CNN','RNN','LSTM']

    # Specify CL strategies
    output_dir = Path(r'C:\Users\jacob\OneDrive\Documents\code\cl code\ehr')

    # Hyperparams for grid search
    config = {'lr':[0.1,0.01,0.001,0.0001], 
              'optimizer':{'SGD':SGD, 'Adam':Adam}, 
              'mem_size':[2,5,10],
              'alpha':[0.001,0.01,0.1,1,10,100,1000],
              'temperature':[0.5,1,1.5,2]}

    models = ['MLP']
    config ={'lr':[0.01], 
              'optimizer':{'SGD':SGD}, 
              'mem_size':[2],
              'alpha':[10],
              'temperature':[1]}

    # Then train over all tasks
    main(data=data, output_dir=output_dir, models=models, config=config)

    # Plotting 
    if False:
        plot_demos()