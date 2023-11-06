import torch
from torch_geometric.nn import SAGEConv, Linear, GCNConv

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        #self.conv1 = GCNConv(-1,hidden_channels,add_self_loops=False)
        #self.conv2 = GCNConv(-1,hidden_channels,add_self_loops=False)
        self.linear = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.linear(x)
        return x