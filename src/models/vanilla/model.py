from src.config.configuration import ConfigurationManager
from torch import optim
from torch import nn
import torch as th


class VanillaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config

        # conv layer
        self.conv_layer = nn.Sequential(
            nn.Conv2d(**self.config.layer_1), # passing layer 1 params as dict
            nn.ReLU(),
            nn.BatchNorm2d(self.config.layer_1.out_channels), # passing output channels
            nn.Conv2d(**self.config.layer_2),
            nn.ReLU(),
            nn.BatchNorm2d(self.config.layer_2.out_channels)
        )

        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(self.get_flattened_size(self.config.image_height, 
                                                    self.config.image_width), 128)
        self.f2 = nn.Linear(128, 8)
        
    def get_flattened_size(self, image_height, image_width):
        # Create a dummy input with batch size 1 and the correct number of input channels
        dummy_input = th.randn(1, self.config.layer_1.in_channels, 
                               image_height, image_width)
        
        # Pass the dummy input through the conv layers
        conv_output = self.conv_layer(dummy_input)
        # Flatten the conv output and calculate its size
        # flattened_size = len(conv_output.view(conv_output.shape[0], -1)[0])
        flattened_size = conv_output.view(1, -1).numel()
        return flattened_size

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = nn.ReLU()(x)
        x = self.f2(x)
        x = nn.ReLU()(x)
        return x
    

    def __name__(self):
        return "Vanilla"

if __name__ == "__main__":
    config_manager = ConfigurationManager()
    config_params = config_manager.get_vanilla_architecture_params()

    model = VanillaModel(config_params)

    # Training loop
    for epoch in range(2):
        model.train()
    
        for images, labels in train_loader:
            # Move images and labels to device (GPU/CPU)
            images, labels = images.to('cpu'), labels.to('cpu')

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            print(f'Epoch [{epoch+1}/{2}], Loss: {loss.item():.4f}')