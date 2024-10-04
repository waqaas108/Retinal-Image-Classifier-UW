import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # (B, C, 1, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        # Average Pooling
        avg_pool = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_pool)
        # Max Pooling
        max_pool = self.max_pool(x).view(b, c)
        max_out = self.fc(max_pool)
        # Combine
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2  # to keep the same spatial size
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Compute average and max along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        out = self.conv(concat)  # (B, 1, H, W)
        out = self.sigmoid(out)  # (B, 1, H, W)
        return x * out.expand_as(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class CNNAttentionImageClassifier(nn.Module):
    def __init__(self, input_size=(3, 256, 256), num_classes=4, num_filters=32, kernel_size=3, 
                 pool_size=2, num_conv_layers=4, attention_reduction=16, attention_kernel_size=7, 
                 fc_size=256, dropout=0.3):
        super(CNNAttentionImageClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.initial_num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_conv_layers = num_conv_layers
        self.attention_reduction = attention_reduction
        self.attention_kernel_size = attention_kernel_size
        self.fc_size = fc_size
        self.dropout = dropout
        
        # Dynamically construct convolutional layers based on the number of layers
        self.convs = nn.ModuleList()
        in_channels = input_size[0]
        current_num_filters = num_filters
        
        for _ in range(num_conv_layers):
            self.convs.append(nn.Conv2d(in_channels, current_num_filters, kernel_size=kernel_size, padding=1))
            self.convs.append(nn.BatchNorm2d(current_num_filters))
            self.convs.append(nn.ReLU())
            self.convs.append(nn.MaxPool2d(kernel_size=pool_size))
            in_channels = current_num_filters
            current_num_filters *= 2  # Double the filters at each layer
        
        # Attention Module
        self.cbam = CBAM(in_channels, reduction=attention_reduction, kernel_size=attention_kernel_size)
        
        # Calculate the feature map size after conv layers
        channels, height, width = self._get_feature_map_size(input_size)
        
        # Fully connected layers for classification with Dropout
        self.fc1 = nn.Linear(channels * height * width, fc_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_size, num_classes)
    
    def _get_feature_map_size(self, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            x = dummy_input
            for layer in self.convs:
                x = layer(x)
            x = self.cbam(x)
            batch_size, channels, height, width = x.size()
            return channels, height, width
    
    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        
        x = self.cbam(x)  # Apply CBAM attention
        
        batch_size, channels, height, width = x.size()
        
        # Flatten the feature maps
        x = x.view(batch_size, channels * height * width)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
    # Example instantiation of the model
    model = CNNAttentionImageClassifier(
        input_size=(3, 256, 256),
        num_classes=4, 
        num_filters=32, 
        kernel_size=3, 
        pool_size=2, 
        num_conv_layers=4, 
        attention_reduction=16, 
        attention_kernel_size=7, 
        fc_size=256, 
        dropout=0.3
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