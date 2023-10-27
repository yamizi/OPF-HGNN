"""
https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html
https://github.com/pyg-team/pytorch_geometric/blob/3752d94447950e83e9d6e03c1332d0c031a2376c/docs/source/tutorial/heterogeneous.rst#L16
"""

from torch_geometric.datasets import OGB_MAG
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn import HeteroConv, GCNConv, GATConv, Linear, SAGEConv, to_hetero
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import train_test_split_edges

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--use-sparse-tensor', type=int,default=0)
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else 'cpu'

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['paper'].train_mask
    loss = F.cross_entropy(out['paper'][mask], data['paper'].y[mask])
    loss.backward()
    optimizer.step()
    return out, float(loss)

def train_with_loader(model, optimizer,train_loader):
    model.train()

    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to('cuda:0')
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = F.cross_entropy(out['paper'][:batch_size],
                               batch['paper'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples

def main():

    transforms = [T.ToUndirected(merge=True)]
    if args.use_sparse_tensor:
        transforms.append(T.ToSparseTensor())

    dataset = OGB_MAG(root='../data', preprocess='metapath2vec',
                    transform=T.Compose(transforms))

    data = dataset[0]
    #data.train_mask = data.val_mask = data.test_mask = None
    #data = train_test_split_edges(data)


    print(data.has_isolated_nodes(),data.has_self_loops(),data.is_undirected())

    
    
    model = GNN(hidden_channels=64, out_channels=dataset.num_classes)
    model = to_hetero(model, data.metadata(), aggr='sum')

    model = model.to(device)

    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #_, l = train(model,optimizer,data)
    #print("one step train loss", l)
    for epoch in range(1, 21):
        print("multistep train epoch",epoch)
        out, loss = train(model,optimizer,data)
        print("train loss",float(loss))

        model.eval()
        test_mask = data['paper'].test_mask
        test_pred = out['paper'][test_mask].max(1)[1]
        test_loss = F.cross_entropy(out['paper'][test_mask], data['paper'].y[test_mask])
        print("test loss",float(test_loss))

    
     # Data Loader for pytorch geometric using data
    
    train_loader = NeighborLoader(
        data,
        # Sample 15 neighbors for each node and each edge type for 2 iterations:
        num_neighbors=[15] * 2,
        # Use a batch size of 128 for sampling training nodes of type "paper":
        batch_size=128,
        input_nodes=('paper', data['paper'].train_mask),
    )

    train_loader = DataLoader([data]*50, batch_size=128)

    batch = next(iter(train_loader))
    print(batch)



if __name__ == "__main__":
    main()