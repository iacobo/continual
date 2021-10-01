"""
PyTorch Neural Network model definitions.

Consists of simple parameterised:

- MLP:         Dense Feedforward ANN      / "Multilayer Perceptron"
- CNN:         1d CNN                     / "Temporal CNN" (TCN)
- RNN:         Recurrent Neural network
- GRU:         Gated Recurrent Unit
- LSTM:        Long-short term memory RNN
- Transformer: 

Models generally of format:

=================================================================
Layer (type:depth-idx)                   Output Shape             
=================================================================
SimpleMLP                                --                      
├─Sequential: 1-1                            
│    └─Sequential: 2-1                   [n, hidden_dim]       
│    │    └─Linear: 3-1                        
│    │    └─Nonlinearity: 3-2                  
│    └─Sequential: 2-2                   [n, hidden_dim]       
│    │    └─Linear: 3-3                              
│    │    └─Non-linearity: 3-4                      
|    |
                      ... (n_layers) ...
|    |
│    └─Sequential: 2-n                   [n, hidden_dim]                
│    │    └─Linear: 3-2n+1                     
│    │    └─Nonlinearity: 3-2n+2          
├─Sequential: 1-2                        [n, hidden_dim//2]  
│    └─Linear: 2-1                       
|    └─Linear: 2-2                       [n, output_size]      
=================================================================

Where the number of layers, layer width, nonlinearity, and degree of dropout are parameterised.

Model specific parameters:

- CNN          Kernel width
- RNN/LSTM/GRU Bidirectionality
- Transformer  Number of heads

"""

from torch import nn

class SimpleMLP(nn.Module):
    """
    Feed-forward network ("multi-layer perceptron")
    """
    def __init__(self, n_channels, seq_len, hidden_dim, n_layers, output_size=2, dropout=0, nonlinearity='relu'):
        super().__init__()

        if nonlinearity == 'relu':
            nonlinearity = nn.ReLU
        elif nonlinearity == 'tanh':
            nonlinearity = nn.Tanh

        layers = []

        for i in range(n_layers):
            if i == 0:
                current_layer = nn.Sequential(
                    nn.Linear(in_features=seq_len*n_channels, out_features=hidden_dim, bias=True),
                    nonlinearity(),
                    nn.Dropout(p=dropout)
                )
            else:
                current_layer = nn.Sequential(
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
                    nonlinearity(),
                    nn.Dropout(p=dropout)
                )
            layers.append(current_layer)

        self.features = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim//2, bias=True),
            nn.Linear(in_features=hidden_dim//2, out_features=output_size, bias=True)
        )

    def forward(self, x):
        """
        Forward pass of model.
        """
        batch_size = x.shape[0]

        out = x.view(batch_size, -1)
        out = self.features(out)
        out = self.fc(out)
        return out


class SimpleRNN(nn.Module):
    """
    RNN
    """
    def __init__(self, n_channels, seq_len, hidden_dim, n_layers, output_size=2, bidirectional=True, nonlinearity='tanh', dropout=0):
        super().__init__()

        scalar = 2 if bidirectional else 1

        self.rnn = nn.RNN(n_channels, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout, nonlinearity=nonlinearity)
        self.fc = nn.Sequential(
            nn.Linear(in_features=scalar*seq_len*hidden_dim, out_features=scalar*seq_len*hidden_dim//2, bias=True),
            nn.Linear(in_features=scalar*seq_len*hidden_dim//2, out_features=output_size, bias=True)
        )

    def forward(self, x):
        """
        Forward pass of model.
        """
        batch_size = x.shape[0]

        out, _ = self.rnn(x)
        out = out.reshape(batch_size, -1)
        out = self.fc(out)
        return out


class SimpleLSTM(nn.Module):
    """
    LSTM
    """
    def __init__(self, n_channels, seq_len, hidden_dim, n_layers, output_size=2, bidirectional=True, dropout=0):
        super().__init__()

        scalar = 2 if bidirectional else 1

        self.lstm = nn.LSTM(n_channels, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(in_features=scalar*seq_len*hidden_dim, out_features=scalar*seq_len*hidden_dim//2, bias=True),
            nn.Linear(in_features=scalar*seq_len*hidden_dim//2, out_features=output_size, bias=True)
        )

    def forward(self, x):
        """
        Forward pass of model.
        """
        batch_size = x.shape[0]

        out, _ = self.lstm(x)
        out = out.reshape(batch_size, -1)
        out = self.fc(out)
        return out


class SimpleGRU(nn.Module):
    """
    GRU
    """
    def __init__(self, n_channels, seq_len, hidden_dim, n_layers, output_size=2, bidirectional=True, dropout=0):
        super().__init__()

        scalar = 2 if bidirectional else 1

        self.lstm = nn.GRU(n_channels, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(in_features=scalar*seq_len*hidden_dim, out_features=scalar*seq_len*hidden_dim//2, bias=True),
            nn.Linear(in_features=scalar*seq_len*hidden_dim//2, out_features=output_size, bias=True)
        )

    def forward(self, x):
        """
        Forward pass of model.
        """
        batch_size = x.shape[0]

        out, _ = self.lstm(x)
        out = out.reshape(batch_size, -1)
        out = self.fc(out)
        return out


class SimpleCNN(nn.Module):
    """
    1d CNN (also known as TCN)

    `kernel_size` must be odd for `padding` to work as expected.
    """
    def __init__(self, n_channels, seq_len, hidden_dim, n_layers, output_size=2, kernel_size=3, nonlinearity='relu'):
        super().__init__()

        if nonlinearity == 'relu':
            nonlinearity = nn.ReLU
        elif nonlinearity == 'tanh':
            nonlinearity = nn.Tanh

        layers = []
        n_pools=0

        for i in range(n_layers):
            in_channels = n_channels if i==0 else hidden_dim

            current_layer = nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size, stride=1, padding=kernel_size//2),
                # JA: Investigate removing BatchNorm as bad for CL
                #nn.BatchNorm1d(hidden_dim),
                nonlinearity()
                )
            layers.append(current_layer)

            # Ensure MaxPools don't wash out entire sequence
            if seq_len // 2**(n_pools+1) > 2:
                n_pools += 1
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.cnn_layers = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_dim * (seq_len // 2**n_pools), out_features=(hidden_dim * (seq_len // 2**n_pools))//2, bias=True),
            nn.Linear(in_features=(hidden_dim * (seq_len // 2**n_pools))//2, out_features=output_size, bias=True)
        )

    def forward(self, x):
        """
        Forward pass of model.
        """
        batch_size = x.shape[0]

        out = x.swapdims(1,2)
        out = self.cnn_layers(out)
        out = out.reshape(batch_size, -1)
        out = self.fc(out)
        return out

class SimpleTransformer(nn.Module):
    """
    Transformer.
    """
    def __init__(self, n_channels, seq_len, hidden_dim, n_layers, n_heads=8, output_size=2, nonlinearity='relu', dropout=0):
        super().__init__()

        # JA: need to make this more elegant
        while seq_len % n_heads != 0:
            n_heads -=1

        transformer_layer = nn.TransformerEncoderLayer(d_model=seq_len, dim_feedforward=hidden_dim, nhead=n_heads, activation=nonlinearity, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)
        self.fc = nn.Linear(seq_len*n_channels, output_size)

    def forward(self, x):
        """
        Forward pass of model.
        """
        batch_size = x.shape[0]

        out = x.swapdims(1,2)
        out = self.transformer(out)
        out = out.reshape(batch_size, -1)
        out = self.fc(out)
        return out

MODELS = {
    'MLP':SimpleMLP,
    'CNN':SimpleCNN,
    'RNN':SimpleRNN,'LSTM':SimpleLSTM,'GRU':SimpleGRU,
    'Transformer':SimpleTransformer
    }
