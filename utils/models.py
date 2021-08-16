from torch import nn

# CL imports
from avalanche.training.strategies import Naive, JointTraining, Cumulative # Baselines
from avalanche.training.strategies import EWC, LwF, SynapticIntelligence   # Regularisation
from avalanche.training.strategies import Replay, GDumb, GEM, AGEM         # Rehearsal
from avalanche.training.strategies import AR1, CWRStar, CoPE, StreamingLDA


class SimpleMLP(nn.Module):
    def __init__(self, n_channels, seq_len, hidden_dim=512, output_size=2, dropout=0, nonlinearity='relu'):
        super().__init__()

        if nonlinearity == 'relu':
            nonlinearity = nn.ReLU
        elif nonlinearity == 'tanh':
            nonlinearity = nn.Tanh

        self.features = nn.Sequential(
            nn.Linear(in_features=seq_len*n_channels, out_features=hidden_dim, bias=True),
            nonlinearity(),
            nn.Dropout(p=dropout),

            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
            nonlinearity(),
            nn.Dropout(p=dropout),

            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
            nonlinearity(),
            nn.Dropout(p=dropout),

            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
            nonlinearity(),
            nn.Dropout(p=dropout),
            )
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_size, bias=True)

    def forward(self, x):
        batch_size = x.shape[0]

        out = x.view(batch_size, -1)
        out = self.features(out)
        out = self.fc(out)
        return out


class SimpleRNN(nn.Module):
    def __init__(self, n_channels, seq_len, hidden_dim=512, n_layers=1, output_size=2, bidirectional=True, nonlinearity='tanh', dropout=0):
        super().__init__()

        scalar = 2 if bidirectional else 1

        self.rnn = nn.RNN(n_channels, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout, nonlinearity=nonlinearity)
        self.fc = nn.Linear(scalar*seq_len*hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.shape[0]

        out, _ = self.rnn(x)
        out = out.reshape(batch_size, -1)
        out = self.fc(out)
        return out


class SimpleLSTM(nn.Module):
    def __init__(self, n_channels, seq_len, hidden_dim=512, n_layers=1, output_size=2, bidirectional=True, dropout=0):
        super().__init__()

        scalar = 2 if bidirectional else 1

        self.lstm = nn.LSTM(n_channels, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(scalar*seq_len*hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.shape[0]

        out, _ = self.lstm(x)
        out = out.reshape(batch_size, -1)
        out = self.fc(out)
        return out

class SimpleCNN(nn.Module):
    def __init__(self, n_channels, seq_len, hidden_dim=512, output_size=2, nonlinearity='relu'):
        super().__init__()

        if nonlinearity == 'relu':
            nonlinearity = nn.ReLU
        elif nonlinearity == 'tanh':
            nonlinearity = nn.Tanh

        self.cnn_layers = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nonlinearity(),
            #nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim//2),
            nonlinearity(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(hidden_dim//2, hidden_dim//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim//4),
            nonlinearity(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear((seq_len//4)*(hidden_dim//4), output_size) #(seq_len//2*num batch norm) * final hid size

    # Defining the forward pass    
    def forward(self, x):
        batch_size = x.shape[0]

        out = x.swapdims(1,2)
        out = self.cnn_layers(out)
        out = out.view(batch_size, -1)
        out = self.fc(out)
        return out

#class SimpleTransformer():
#
#    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# CONTAINERS
MODELS = {'MLP':SimpleMLP, 'CNN':SimpleCNN, 'RNN':SimpleRNN, 'LSTM':SimpleLSTM}
STRATEGIES = {'Naive':Naive, 'Joint':JointTraining, 'Cumulative':Cumulative,
              'EWC':EWC, 'LwF':LwF, 'SI':SynapticIntelligence, 
              'Replay':Replay, 'GEM':GEM, 'AGEM':AGEM, 'GDumb':GDumb, 'CoPE':CoPE, 
              'AR1':AR1, 'StreamingLDA':StreamingLDA, 'CWRStar':CWRStar}