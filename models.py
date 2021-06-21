import torch
from torch import nn
from avalanche.models import SimpleMLP as avaMLP

class SimpleMLP(avaMLP):
    def __init__(self, input_size, n_channels, output_size=2):
        super(SimpleMLP, self).__init__()

        self.mlp = avaMLP(input_size=input_size*n_channels, num_classes=output_size)

    def forward(self, x):
        x = self.mlp(x)
        return x


class SimpleRNN(nn.Module):
    def __init__(self, input_size, n_channels, output_size=2, hidden_dim=10, n_layers=2):
        super(SimpleRNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(n_channels*hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(batch_size, -1)
        out = self.fc(out)
        
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

class SimpleLSTM(nn.Module):

    def __init__(self, input_size, n_channels, hidden_dim=10, output_size=2):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(n_channels*hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        x, _ = self.lstm(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x

class SimpleCNN(nn.Module):   
    def __init__(self, n_channels):
        super(SimpleCNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv1d(n_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv1d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(8, 2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = x.swapdims(-1,-2) # Change from NLC to NCL format
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
