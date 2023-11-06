import torch
from torch_geometric.nn import SAGEConv, Linear, GCNConv
from torch.nn import Linear as Linear2d

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.linear = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.linear(x)
        return x

class FCNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = Linear2d(-1, hidden_channels)
        self.conv2 = Linear2d(hidden_channels, hidden_channels)
        self.linear = Linear2d(hidden_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        x = x.relu()
        x = self.linear(x)
        return x