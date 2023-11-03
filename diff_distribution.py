import uuid
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
sys.path.append("../")

from utils.logging import init_comet, log_dict_series, log_opf
from utils.pandapower import build_dataset
from utils.pandapower.opf_validation import validate_opf
import pandapower as pp
from torch_geometric.nn import to_hetero

from torch_geometric.loader import NeighborLoader, DataLoader
import torch
from utils.base_gnn import GNN

from utils.train import train_opf
from utils.plot import plot_losses, plot_results
import json

def run_case(training_cases=[["case9",64,0.7,["cost", "load"]]],experiment=None,
             validation_case=["case9",64,0.7,["cost", "load"]] ,plot=True,
             save_path="./output", title="",dataset_type="y_no_OPF",
             max_epochs=200, y_nodes=["gen","ext_grid"], train_batch_size=5,val_batch_size=5):
    
    uniqueid = uuid.uuid4()
    if experiment is not None:
        experiment.log_parameters({"uniqueid":uniqueid, "max_epochs":max_epochs,"dataset_type":dataset_type,
                               "save_path":save_path,"title":title,"y_nodes":y_nodes,"plot":plot})
    
    val_case_name, nb_graphs, mutation_rate, mutations = validation_case
    val_graphs, valid_networks, _, _ = build_dataset(val_case_name,nbsamples=nb_graphs,save_dataframes=save_path,
                                               mutation_rate=mutation_rate, uniqueid="{}/val".format(uniqueid),
                                               dataset_type=dataset_type, experiment=experiment, mutations=mutations)
    print("Correct validation graphs {}/{}".format(len(val_graphs),nb_graphs))
    val_loader = DataLoader([g[0] for g in val_graphs], batch_size=train_batch_size)
    experiment.log_metric("nb_valid_graphs", len(val_graphs))

    train_graphs = []
    nb_graphs = 0
    for training_case in training_cases:
        train_case_name, nb_graph, mutation_rate, mutations = training_case

        train_graph, _, _, _ = build_dataset(train_case_name,nbsamples=nb_graph,save_dataframes=save_path,
                                               mutation_rate=mutation_rate, uniqueid="{}/train".format(uniqueid),
                                               dataset_type=dataset_type, experiment=experiment, mutations=mutations)
    
        train_graphs += train_graph
        nb_graphs+=nb_graph
    
    print("Correct training graphs {}/{}".format(len(train_graphs),nb_graphs))
    experiment.log_metric("nb_valid_graphs", len(train_graphs))
    train_loader = DataLoader([g[0] for g in train_graphs], batch_size=val_batch_size)

    if len(train_graphs)==0:
        return 
    
    graph_y = train_graphs[0]
    data = graph_y[0]
    model = GNN(hidden_channels=64, out_channels=graph_y.num_outputs)
    model = to_hetero(model, data.metadata(), aggr='sum')
    
    train_losses, val_losses, val_losses_gen, val_losses_ext_grid, last_out = train_opf(model,train_loader,val_loader, max_epochs=max_epochs, y_nodes=y_nodes)
    constrained_networks, errors_network = validate_opf(valid_networks, val_graphs, last_out, y_nodes=y_nodes)
    
    case_name = "{}->{}".format(train_case_name,val_case_name)

    log_dict = {"constraint":constrained_networks,"train_losses":train_losses, "val_losses":val_losses, "val_losses_gen":val_losses_gen, "val_losses_ext_grid":val_losses_ext_grid}
    with open(save_path+"/losses.json", "w") as outfile:
        json.dump(log_dict, outfile)
    
    if experiment is not None:
        log_dict_series(log_dict, experiment)
        log_opf(val_graphs, last_out, y_nodes,experiment)

    
    
    if plot:
        plot_losses(train_losses,val_losses,val_losses_gen, val_losses_ext_grid, case_name, title, save_path)
        plot_results(valid_networks, val_graphs, last_out, y_nodes, constrained_networks, errors_network, case_name, title, save_path)


if __name__ == "__main__":

    training_case=[["case9",32,0.7,["cost"]]]
    validation_case=["case9",8,0.7,["cost"]]
    
    experiment = init_comet({"cases":"case9"})
    run_case(training_cases=training_case,validation_case=validation_case, 
            title="generalization cost", save_path="./output/case9_9", max_epochs=5, experiment=experiment)
    plt.show()
    exit()

    training_case=[["case9",64,0.7,["cost"]]]
    validation_case=["case14",32,0.7,["cost"]]
    run_case(training_cases=training_case,validation_case=validation_case, 
            title="generalization cost", save_path="./output/case9_14")
