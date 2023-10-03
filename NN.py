from torch import nn
import torch

device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')


class ResBlock(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=False),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Droput(p=0.6)
        )

    def forward(self, x):
        residue = x
        return self.fc(self.fc(x)) + residue


class MLOptimizer(nn.Module):
    def __init__(self, num_layers, h_dim, n_classes, n_features):
        super().__init__()
        modules = []

        modules.extend([nn.Linear(n_features, h_dim, bias=False),
                        nn.BatchNorm1d(h_dim),
                        nn.ReLU(),
                        nn.Dropout(p=0.0)])

        for _ in range(num_layers // 2 - 1):
            modules.append(ResBlock(h_dim))

        modules.append(nn.Linear(h_dim, n_classes, bias=True))

        self.hidden_layers = nn.Sequential(*modules)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = nn.Softmax(x)
        return x
    


