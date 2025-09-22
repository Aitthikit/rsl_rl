import torch
import torch.nn as nn

class ObsEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], output_dim=8):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        
        # Final layer
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, gru_hidden_size=256, gru_num_layers=2, hidden_dims=[256, 128], output_dim=8, bidirectional=False):
        super().__init__()
        
        self.hidden_dim = gru_hidden_size
        self.num_layers = gru_num_layers
        self.num_directions = 2 if bidirectional else 1
        
        # Initial dense layer to process input
        self.input_layer = nn.Linear(input_dim, gru_hidden_size)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Build MLP layers after GRU
        mlp_layers = []
        dims = [gru_hidden_size * self.num_directions] + hidden_dims
        
        for i in range(len(dims)-1):
            mlp_layers.append(nn.Linear(dims[i], dims[i+1]))
            mlp_layers.append(nn.ReLU())
            
        # Final output layer
        mlp_layers.append(nn.Linear(dims[-1], output_dim))
        
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
        # Hidden state
        self.hidden = None
        
    def forward(self, x, hidden=None):
        # Reshape input if needed [batch, features] -> [batch, 1, features]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Process through input layer
        x = torch.relu(self.input_layer(x))
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
            
        # GRU forward pass
        output, hidden = self.gru(x, hidden)
        
        # Use last output for final prediction
        last_output = output[:, -1]
        
        # Forward through MLP layers
        encoded = self.mlp_layers(last_output)
        
        # Store hidden state
        self.hidden = hidden
        
        return encoded
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )
    
    def reset(self, dones=None):
        """Reset hidden states"""
        if dones is None:
            self.hidden = None
        else:
            if self.hidden is not None:
                self.hidden[:, dones] = 0.0

class ConvGRUEncoder(nn.Module):
    def __init__(self, input_shape, conv_channels=[32, 64, 128], conv_kernel_sizes=[3, 3, 3],
                 conv_strides=[2, 2, 2], gru_hidden_size=256, gru_num_layers=1, 
                 hidden_dims=[256, 128], output_dim=8):
        super().__init__()
        
        # Image input shape (channels, height, width)
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        
        # Define convolutional layers
        conv_layers = []
        in_channels = self.input_channels
        
        for out_channels, kernel_size, stride in zip(conv_channels, conv_kernel_sizes, conv_strides):
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                         stride=stride, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            ])
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate the size of flattened features after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
            conv_out = self.conv_layers(dummy_input)
            self.conv_flat_size = conv_out.numel() // conv_out.size(0)
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=self.conv_flat_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True
        )
        
        # Build MLP layers after GRU
        mlp_layers = []
        dims = [gru_hidden_size] + hidden_dims
        
        for i in range(len(dims)-1):
            mlp_layers.append(nn.Linear(dims[i], dims[i+1]))
            mlp_layers.append(nn.ReLU())
            
        # Final output layer
        mlp_layers.append(nn.Linear(dims[-1], output_dim))
        
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
        # Hidden state
        self.hidden = None
        self.gru_num_layers = gru_num_layers
        self.gru_hidden_size = gru_hidden_size
        
    def forward(self, x, hidden=None):
        # Input shape should be [batch_size, sequence_length, channels, height, width]
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Reshape for conv layers
        x = x.view(-1, self.input_channels, self.input_height, self.input_width)
        
        # Forward through conv layers
        conv_out = self.conv_layers(x)
        
        # Reshape for GRU: [batch_size, sequence_length, features]
        conv_flat = conv_out.view(batch_size, seq_length, -1)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
            
        # GRU forward pass
        output, hidden = self.gru(conv_flat, hidden)
        
        # Use last output for final prediction
        last_output = output[:, -1]
        
        # Forward through MLP layers
        encoded = self.mlp_layers(last_output)
        
        # Store hidden state
        self.hidden = hidden
        
        return encoded
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(
            self.gru_num_layers,
            batch_size,
            self.gru_hidden_size,
            device=device
        )
    
    def reset(self, dones=None):
        """Reset hidden states"""
        if dones is None:
            self.hidden = None
        else:
            if self.hidden is not None:
                self.hidden[:, dones] = 0.0
        
class ConvEncoder(nn.Module):
    def __init__(self, input_dim, conv_channels=[32, 64, 128], conv_kernel_sizes=[3, 3, 3], 
                 conv_strides=[1, 1, 1], hidden_dims=[256, 128], output_dim=8):
        super().__init__()
        
        # Reshape input into pseudo-image format
        self.input_channels = 1  # Treat features as a 1D signal
        self.input_height = int(input_dim ** 0.5)  # Square root to make a square image
        self.input_width = self.input_height
        if self.input_height * self.input_width < input_dim:
            self.input_height += 1
        
        # Define convolutional layers
        conv_layers = []
        in_channels = self.input_channels
        
        for out_channels, kernel_size, stride in zip(conv_channels, conv_kernel_sizes, conv_strides):
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                         stride=stride, padding=kernel_size//2),
                nn.ReLU()
            ])
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate the size of flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
            conv_out = self.conv_layers(dummy_input)
            flattened_size = conv_out.numel() // conv_out.size(0)
        
        # Define fully connected layers
        fc_layers = []
        dims = [flattened_size] + hidden_dims
        
        for i in range(len(dims)-1):
            fc_layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU()
            ])
        
        # Final output layer
        fc_layers.append(nn.Linear(dims[-1], output_dim))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
    def forward(self, x):
        # If input is not 4D, reshape it
        if len(x.shape) != 4:
            # Assume input is [batch, features]
            # Reshape to [batch, channels, height, width]
            batch_size = x.size(0)
            x = x.view(batch_size, 1, -1, 1)
            
        # Forward through conv layers
        conv_out = self.conv_layers(x)
        
        # Flatten
        flattened = conv_out.view(conv_out.size(0), -1)
        
        # Forward through fc layers
        encoded = self.fc_layers(flattened)
        
        return encoded
