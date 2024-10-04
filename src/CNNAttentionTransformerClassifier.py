import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class CNNAttentionTransformerClassifier(nn.Module):
    def __init__(self, input_size=(3, 256, 256), num_classes=4, num_filters=32, kernel_size=3, 
                 pool_size=2, num_conv_layers=4, transformer_dim=256, nhead=4, num_transformer_layers=3, 
                 dim_feedforward=512, dropout=0.3, fc_size=256):
        super(CNNAttentionTransformerClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.initial_num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_conv_layers = num_conv_layers
        self.transformer_dim = transformer_dim
        self.nhead = nhead
        self.num_transformer_layers = num_transformer_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.fc_size = fc_size
        
        # Dynamically construct convolutional layers based on the number of layers
        self.convs = nn.ModuleList()
        in_channels = input_size[0]
        current_num_filters = num_filters
        
        for _ in range(num_conv_layers):
            self.convs.append(nn.Conv2d(in_channels, current_num_filters, kernel_size=kernel_size, padding=1))
            self.convs.append(nn.ReLU())
            self.convs.append(nn.MaxPool2d(kernel_size=pool_size))
            in_channels = current_num_filters
            current_num_filters *= 2  # Double the filters at each layer
        
        # Calculate the feature map size after conv layers
        channels, height, width = self._get_feature_map_size(input_size)
        
        # Projection layer to match transformer_dim
        self.proj = nn.Linear(channels, transformer_dim)  # Project channels to transformer_dim
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=transformer_dim)
        
        # Transformer Encoder with batch_first=True and smaller dimension/feedforward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True  # Enable batch_first to align input format
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Classification head
        self.fc1 = nn.Linear(transformer_dim, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)
        self.dropout_layer = nn.Dropout(dropout)
    
    def _get_feature_map_size(self, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            x = dummy_input
            for layer in self.convs:
                x = layer(x)
            batch_size, channels, height, width = x.size()
            return channels, height, width
    
    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        
        batch_size, channels, height, width = x.size()
        
        # Reshape to (batch_size, seq_length=height*width, channels)
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch_size, height, width, channels)
        x = x.view(batch_size, height * width, channels)
        
        # Project channels to transformer dimension
        x = self.proj(x)  # (batch_size, seq_length, transformer_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # (batch_size, seq_length, transformer_dim)
        
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_length, transformer_dim)
        
        # Aggregate Transformer outputs using Global Average Pooling
        x = x.mean(dim=1)  # (batch_size, transformer_dim)
        
        # Classification head
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
    # Example instantiation of the model with aggregation
    model = CNNAttentionTransformerClassifier(
        input_size=(3, 256, 256),
        num_classes=4, 
        num_filters=32, 
        kernel_size=3, 
        pool_size=2, 
        num_conv_layers=4, 
        transformer_dim=256, 
        nhead=4, 
        num_transformer_layers=3, 
        dim_feedforward=512, 
        dropout=0.3, 
        fc_size=256
    )
    
    # Create random input tensor with batch size of 8 and updated image size
    input_tensor = torch.randn(8, 3, 256, 256)
    
    # Move model and input to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Forward pass to check the output size
    output = model(input_tensor)
    print(f"Output shape: {output.size()}")  # Should be (8, 4)
