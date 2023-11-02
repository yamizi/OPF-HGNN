import uuid
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

from utils.train import train_opf
from utils.plot import plot_losses
import json

def run_case(training_cases=[["case9",64,0.7,["cost", "load"]]],
             validation_case=["case9",64,0.7,["cost", "load"]] ,
             save_path="./output", title="",dataset_type="y_no_OPF"):
    
    uniqueid = uuid.uuid4()
    
    val_case_name, nb_graphs, mutation_rate, mutations = validation_case
    val_graphs, network, _, _ = build_dataset(val_case_name,nbsamples=nb_graphs,save_dataframes=save_path,
                                               mutation_rate=mutation_rate, mutations=mutations, uniqueid=uniqueid,
                                               dataset_type=dataset_type)
    print("Correct validation graphs {}/{}".format(len(val_graphs),nb_graphs))
    val_loader = DataLoader([g[0] for g in val_graphs], batch_size=5)


    train_graphs = []
    nb_graphs = 0
    for training_case in training_cases:
        train_case_name, nb_graph, mutation_rate, mutations = training_case

        train_graph, _, _, _ = build_dataset(train_case_name,nbsamples=nb_graph,save_dataframes=save_path,
                                               mutation_rate=mutation_rate, mutations=mutations, uniqueid=uniqueid
                                               ,dataset_type=dataset_type)
    
        train_graphs += train_graph
        nb_graphs+=nb_graph
    
    print("Correct training graphs {}/{}".format(len(train_graphs),nb_graphs))
    train_loader = DataLoader([g[0] for g in train_graphs], batch_size=5)

    if len(train_graphs)==0:
        return 
    
    graph_y = train_graphs[0]
    data = graph_y[0]
    model = GNN(hidden_channels=64, out_channels=graph_y.num_outputs)
    model = to_hetero(model, data.metadata(), aggr='sum')
    
    train_losses, val_losses, val_losses_gen, val_losses_ext_grid = train_opf(model,train_loader,val_loader)
    case_name = "{}->{}".format(train_case_name,val_case_name)

    with open(save_path+"/losses.json", "w") as outfile:
        json.dump({"train_losses":train_losses, "val_losses":val_losses, "val_losses_gen":val_losses_gen, "val_losses_ext_grid":val_losses_ext_grid}, outfile)
    plot_losses(train_losses,val_losses,val_losses_gen, val_losses_ext_grid, case_name, title, save_path)



training_case=[["case9",4,0.7,["load_relative"]]]
validation_case=["case9",4,0.7,["cost"]]
run_case(training_cases=training_case,validation_case=validation_case, title="generalization cost", save_path="./output/case9_14")



training_case=[["case9",64,0.7,["cost"]]]
validation_case=["case14",32,0.7,["cost"]]
run_case(training_cases=training_case,validation_case=validation_case, title="generalization cost", save_path="./output/case9_14")
