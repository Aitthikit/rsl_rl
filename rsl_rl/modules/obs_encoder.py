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
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=8, bidirectional=False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        # Initial dense layer to process input
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output layer
        gru_output_dim = hidden_dim * self.num_directions
        self.output_layer = nn.Linear(gru_output_dim, output_dim)
        
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
        encoded = self.output_layer(last_output)
        
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

class ConvEncoder(nn.Module):
    def __init__(self, input_channels, input_height, input_width, output_dim=8):
        super().__init__()
        
        # Define convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Third conv block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Calculate the size of flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            conv_out = self.conv_layers(dummy_input)
            flattened_size = conv_out.numel() // conv_out.size(0)
        
        # Define fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
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
