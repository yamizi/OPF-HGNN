import torch
from torch_geometric.nn import SAGEConv, Linear, GCNConv
from torch.nn import Linear as Linear2d
from collections import OrderedDict


class GNN(torch.nn.Module):
    def __init__(self, initial_channels, hidden_channels, nb_hidden_layers, out_channels):
        super().__init__()

        self.first_conv = SAGEConv((-1, -1), initial_channels)
        self.convs = torch.nn.ModuleDict(
            OrderedDict([(f"conv{i}", SAGEConv((-1, -1), hidden_channels)) for i in range(nb_hidden_layers)]))
        self.linear = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.first_conv(x, edge_index)
        x = x.relu()
        for (k, v) in self.convs.items():
            x = v(x, edge_index)
            x = x.relu()

        x = self.linear(x)
        return x


class FCNN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = Linear2d(input_channels, hidden_channels)
        self.conv2 = Linear2d(hidden_channels, hidden_channels)
        self.linear = Linear2d(hidden_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        x = x.relu()
        x = self.linear(x)
        return x
