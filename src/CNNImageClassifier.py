import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNImageClassifier(nn.Module):
    def __init__(self, input_size=(3, 256, 256), num_classes=4, num_filters=32, kernel_size=3, pool_size=2, num_layers=4, fc_size=512):
        super(CNNImageClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.initial_num_filters = num_filters  # Keep the initial number of filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_layers = num_layers
        self.fc_size = fc_size
        
        # Dynamically construct convolutional layers based on the number of layers
        self.convs = nn.ModuleList()
        in_channels = input_size[0] 
        current_num_filters = num_filters  # To keep track of the number of filters
        
        for _ in range(num_layers):
            self.convs.append(nn.Conv2d(in_channels, current_num_filters, kernel_size=kernel_size, padding=0))  # No padding
            self.convs.append(nn.ReLU())
            self.convs.append(nn.MaxPool2d(kernel_size=pool_size))
            in_channels = current_num_filters
            current_num_filters *= 2  # Double the filters at each layer
        
        # Calculate the flattened size by passing a dummy input through the conv layers
        self.flattened_size = self._get_flattened_size(input_size)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(self.flattened_size, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)
    
    def _get_flattened_size(self, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            x = dummy_input
            for layer in self.convs:
                x = layer(x)
            return x.view(1, -1).size(1)
    
    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
    # Example instantiation of the model
    model = CNNImageClassifier(
        input_size=(3, 256, 256),
        num_classes=4, 
        num_filters=32, 
        kernel_size=3, 
        pool_size=2, 
        num_layers=4, 
        fc_size=512
    )
    
    # Create random input tensor with batch size of 8 and updated image size
    input_tensor = torch.randn(8, 3, 256, 256)
    
    # Forward pass to check the output size
    output = model(input_tensor)
    print(f"Output shape: {output.size()}")  # Should be (8, 4)