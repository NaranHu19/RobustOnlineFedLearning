# PyTorch Imports
import torch.nn.functional as F
import torch.nn as nn

##### FNN Class #####

class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x