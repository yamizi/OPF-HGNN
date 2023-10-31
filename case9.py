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

def run_case(case_name="case9", nb_graphs = 64, save_path="./output", 
             mutation_rate=0.7,mutations = ["cost", "load"], title=""):
    
    graphs, network, save_path = build_dataset(case_name,nbsamples=nb_graphs,save_dataframes=save_path,
                                               mutation_rate=mutation_rate, mutations=mutations)
    print("valid graphs {}/{}".format(len(graphs),nb_graphs))
    split_index = int(len(graphs)*3/4)

    if len(graphs)==0:
        return 
    
    graph_y = graphs[0]
    data = graph_y[0]
    model = GNN(hidden_channels=64, out_channels=graph_y.num_outputs)
    model = to_hetero(model, data.metadata(), aggr='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    train_loader = DataLoader([g[0] for g in graphs[:split_index]], batch_size=5)
    val_loader = DataLoader([g[0] for g in graphs[split_index:]], batch_size=5)
    train_losses = []
    val_losses = []
    val_losses_gen = []
    val_losses_ext_grid  = []

    for epoch in range(1,200):
        train_loss = 0
        print("epoch",epoch)
        for batch in train_loader:
            out, loss, losses = train_step(model, optimizer,batch,None,["gen","ext_grid"],torch.nn.L1Loss())
            train_loss += loss

        train_loss /= len(train_loader)
        print("training loss",train_loss)
        train_losses.append(train_loss)

        val_loss = 0
        val_loss_gen = 0
        val_loss_ext_grid = 0
        
        for batch in val_loader:
            out, loss, losses = eval_step(model, batch,None,["gen","ext_grid"],torch.nn.L1Loss())
            val_loss += loss
            val_loss_gen += losses[0]
            val_loss_ext_grid += losses[1]

        val_loss /= len(val_loader)
        print("validation loss",val_loss)
        val_losses.append(val_loss)
        
        val_loss_gen /= len(val_loader)
        val_losses_gen.append(val_loss_gen)

        val_loss_ext_grid /= len(val_loader)
        val_losses_ext_grid.append(val_loss_ext_grid)


    fig = plt.figure()
    plt.figure().clear()

    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="validation loss")
    plt.plot(val_losses_gen, label="validation loss generators")
    plt.plot(val_losses_ext_grid, label="validation loss ext_grid")

    plt.xlabel("training epoch")
    plt.ylabel("L1 error")
    plt.title('Learning p_mw and q_mvar for generators on '+case_name+" "+title)
    plt.legend()
    plt.savefig(save_path+"/losses.png")
    print("over")

    plt.close()
    plt.cla()
    plt.clf()

run_case(case_name="case9", nb_graphs = 64,mutations = ["load"], title="load mutation", save_path="./output/case9")
run_case(case_name="case9", nb_graphs = 64,mutations = ["cost"], title="cost mutation", save_path="./output/case9")
run_case(case_name="case9", nb_graphs = 64,mutations = ["load", "cost"], title="load & cost mutation", save_path="./output/case9")
#run_case(case_name="case14", nb_graphs = 64)
#run_case(case_name="case118", nb_graphs = 64)
#run_case(case_name="case300", nb_graphs = 64)
#run_case(case_name="example_multivoltage", nb_graphs = 64, mutation_rate=0)