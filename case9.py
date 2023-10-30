import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
sys.path.append("../")

from utils.pandapower import PandaPowerDataset, build_costs
import pandapower as pp
from torch_geometric.nn import to_hetero

import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader, DataLoader
import torch
import torch.nn.functional as F
from utils.base_gnn import GNN
from utils.train import train_step


from pandapower.plotting import simple_plot
#simple_plot(network, plot_loads=True)


def build_dataset(nbsamples=20, dataset_type="y_no_OPF"):
    print("building dataset with {nbsamples} variants")
    network = pp.networks.case9()
    graph = PandaPowerDataset(network)
    
    transforms = [T.ToUndirected(merge=True)]
    graphs = []

    if nbsamples==0:
        pp.runopp(network, delta=1e-16)
        graph_y = PandaPowerDataset(network,include_res=False,opf_as_y=True, preprocess='metapath2vec',
                            transform=T.Compose(transforms))
        return [graph_y]
    
    for _ in range(nbsamples):
        costs = [("ext_grid",0,{"cp1_eur_per_mw":np.random.randint(10,100)}),
                          ("gen",0,{"cp1_eur_per_mw":np.random.randint(10,100)})
                        ,("gen",1,{"cp1_eur_per_mw":np.random.randint(10,100)})]
        build_costs(network,costs )
        
        pp.runopp(network, delta=1e-16)

        if dataset_type=="y_no_OPF":
            graph_y = PandaPowerDataset(network,include_res=False,opf_as_y=True, preprocess='metapath2vec',
                            transform=T.Compose(transforms))
        elif dataset_type=="y_OPF":
            graph_y = PandaPowerDataset(network,include_res=True,opf_as_y=True, preprocess='metapath2vec',
                            transform=T.Compose(transforms))
        if dataset_type=="no_y_OPF":
            graph_y = PandaPowerDataset(network,include_res=True,opf_as_y=False, preprocess='metapath2vec',
                            transform=T.Compose(transforms))
            
        graphs.append(PandaPowerDataset(network,preprocess='metapath2vec'))
   
    return graphs, network

graphs, network = build_dataset(5)

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


loader = DataLoader([g[0] for g in graphs], batch_size=5)
for epoch in range(1,100):
    epoch_loss = 0
    for batch in loader:
        print("batch train epoch",epoch, batch)
        out, loss = train_step(model, optimizer,batch,None,"gen",torch.nn.L1Loss())
        epoch_loss += loss

    epoch_loss /= len(loader)
    print("epoch loss",epoch_loss)