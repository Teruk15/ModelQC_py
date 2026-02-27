import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    
    def __init__(self, num_classes: int = 2):
        # Inherit from parent class: nn.Module -> self
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=50,
            kernel_size=(3,50),
            stride=(1,10),
            padding=(0,0),
            bias=True
        )
        
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        
        self.conv2 = nn.Conv2d(
            in_channels=50,
            out_channels=50,
            kernel_size=(1,150),
            stride=(1,1),
            padding=(0,0),
            bias=True
        )
        
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        
        self.drop = nn.Dropout(p=0.5)
        
        # Run dummy forwarding to obtain final-dimension/flatten-layer of features
        # This is required to define FC layers
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 3, 4800)
            feat = self._forward_features(dummy)
            flat_dim = feat.shape[1]
        
        self.fc1 = nn.Linear(flat_dim, 150)
        
        self.fc2 = nn.Linear(150, 25)
        
        self.fc3 = nn.Linear(25, num_classes)
    
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop(x)
        
        # Flatten layer (N, flat_dim)
        x = torch.flatten(x, start_dim=1) 
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        logits = self.fc3(x)
        
        return logits #(N, num_classes)