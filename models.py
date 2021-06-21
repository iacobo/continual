import torch
from torch import nn
from avalanche.models import SimpleMLP as avaMLP

class SimpleMLP(avaMLP):
    def __init__(self, seq_len, n_channels, output_size=2):
        super(SimpleMLP, self).__init__()

        self.mlp = avaMLP(input_size=seq_len*n_channels, num_classes=output_size)

    def forward(self, x):
        out = self.mlp(x)
        return out


class SimpleRNN(nn.Module):
    def __init__(self, n_channels, seq_len, hidden_dim=10, n_layers=2, output_size=2):
        super(SimpleRNN, self).__init__()

        self.rnn = nn.RNN(n_channels, hidden_dim, n_layers, batch_first=True)   
        self.fc = nn.Linear(seq_len*hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.shape[0]

        out, _ = self.rnn(x)
        out = out.reshape(batch_size, -1)
        out = self.fc(out)
        return out

class SimpleLSTM(nn.Module):

    def __init__(self, n_channels, seq_len, hidden_dim=10, n_layers=2, output_size=2):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(n_channels, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(seq_len*hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.shape[0]

        out, _ = self.lstm(x)
        out = out.reshape(batch_size, -1)
        out = self.fc(out)
        return out

class SimpleCNN(nn.Module):   
    # NEED TO SORT OUT MAGIC NUMBERS
    def __init__(self, n_channels, hidden_channels=4, n_output=2):
        super(SimpleCNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv1d(n_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(hidden_channels, n_output)
        )

    # Defining the forward pass    
    def forward(self, x):
        print(x.shape)
        out = x.swapdims(1,2)
        print(out.shape)
        out = self.cnn_layers(out)
        out = out.view(out.size(0), -1)
        out = self.linear_layers(out)
        return out
