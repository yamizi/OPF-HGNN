import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
sys.path.append("../")

from utils.pandapower import PandaPowerDataset
import pandapower as pp
from torch_geometric.nn import to_hetero

import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader, DataLoader
import torch
import torch.nn.functional as F
from utils.base_gnn import GNN
from utils.train import train_step

network = pp.networks.case9()

#build_costs(network, [("ext_grid",0,10),("gen",0,10),("gen",1,10)])
graph = PandaPowerDataset(network)

pp.runopp(network, delta=1e-16)
transforms = [T.ToUndirected(merge=True)]
graph_y = PandaPowerDataset(network,include_res=False,opf_as_y=True, preprocess='metapath2vec',
                    transform=T.Compose(transforms))
graph_opp_y = PandaPowerDataset(network,include_res=True,opf_as_y=True, preprocess='metapath2vec',
                    transform=T.Compose(transforms))
graph_opp = PandaPowerDataset(network,include_res=True,opf_as_y=False, preprocess='metapath2vec',
                    transform=T.Compose(transforms))


from pandapower.plotting import simple_plot
#simple_plot(network, plot_loads=True)

data = graph_y[0]
print(data.has_isolated_nodes(),data.has_self_loops(),data.is_undirected())

model = GNN(hidden_channels=64, out_channels=graph_y.num_outputs)
model = to_hetero(model, data.metadata(), aggr='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for a in range(1,21):
    print("multistep train epoch",a)
    out, loss = train_step(model, optimizer,data,None,"gen",torch.nn.L1Loss())
    print(loss)
