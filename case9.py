import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
sys.path.append("../")

from utils.pandapower import build_dataset
import pandapower as pp
from torch_geometric.nn import to_hetero

from torch_geometric.loader import NeighborLoader, DataLoader
import torch
from utils.base_gnn import GNN
from utils.train import train_step, eval_step
from matplotlib import pyplot as plt

from pandapower.plotting import simple_plot
#simple_plot(network, plot_loads=True)


nb_graphs = 50
split_index = int(nb_graphs*3/4)
graphs, network = build_dataset(nbsamples=nb_graphs)

graph_y = graphs[0]
data = graph_y[0]
print(data.has_isolated_nodes(),data.has_self_loops(),data.is_undirected())

model = GNN(hidden_channels=64, out_channels=graph_y.num_outputs)
model = to_hetero(model, data.metadata(), aggr='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for a in range(1,10):
    print("single graph multistep train epoch",a)
    out, loss = train_step(model, optimizer,data,None,"gen",torch.nn.L1Loss())
    print(loss)

train_loader = DataLoader([g[0] for g in graphs[:split_index]], batch_size=5)
val_loader = DataLoader([g[0] for g in graphs[split_index:]], batch_size=5)
train_losses = []
val_losses = []


for epoch in range(1,200):
    train_loss = 0
    print("epoch",epoch)
    for batch in train_loader:
        out, loss = train_step(model, optimizer,batch,None,"gen",torch.nn.L1Loss())
        train_loss += loss

    train_loss /= len(train_loader)
    print("training loss",train_loss)
    train_losses.append(train_loss)

    val_loss = 0
    for batch in val_loader:
        out, loss = eval_step(model, batch,None,"gen",torch.nn.L1Loss())
        val_loss += loss

    val_loss /= len(val_loader)
    print("validation loss",val_loss)
    val_losses.append(val_loss)

plt.plot(train_losses, label="train loss")
plt.plot(val_losses, label="validation loss")
plt.xlabel("training epoch")
plt.ylabel("L1 error")
plt.title('Learning p_mw and q_mvar for generators on Case9 (2 gen + ext_src)')
plt.show()