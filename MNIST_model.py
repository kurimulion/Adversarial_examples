import torch
import torch.nn as nn
import torch.nn.functional as F

# MNIST model used in "Towards Deep Learning Models Resistant to Adversarial Attacks" arXiv:1706.06083v3

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        
        self.c1 = nn.Conv2d(1, 32, 5, padding=2)
        
        self.c2 = nn.Conv2d(32, 64, 5, padding=2)
        
        self.f1 = nn.Linear(64 * 7 * 7, 1024)
        
        self.f2 = nn.Linear(1024, 10)
        

    def forward(self, x):
        # conv1
        x = self.c1(x)
        x = F.relu(F.max_pool2d(x, 2))
        
        # conv2
        x = self.c2(x)
        x = F.relu(F.max_pool2d(x, 2))

        # fc1
        x = torch.flatten(x, 1)
        x = F.relu(self.f1(x))
        
        # output
        x = self.f2(x)
        
        return x
