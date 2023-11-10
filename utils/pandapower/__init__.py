from typing import Callable, Optional
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset
)
from pandapower.auxiliary import pandapowerNet
import pandapower as pp

import numpy as np

from copy import deepcopy

import torch_geometric.transforms as T
import json
import uuid
import os
from utils.pandapower.mutations import mutate_costs, mutate_loads
import itertools

from utils.pandapower.pandapower_graph import PandaPowerGraph

def clear_duplicates(train_graphs, train_networks, val_graphs, valid_networks):
    print("clearing duplicates")
    train_graphs_c = deepcopy(train_graphs)

    val_str = [val_graph.data.to_dict().__str__() for val_graph in val_graphs]
    train_str = [train_graph.data.to_dict().__str__() for train_graph in train_graphs_c]

    comparisons = np.array([a==b for (a,b) in itertools.product(val_str, train_str)]).reshape(len(val_str),len(train_str))

    nb_duplicates = np.sum(comparisons)
    if nb_duplicates>0:
        print("Found ",nb_duplicates," duplicates")

    return train_graphs, train_networks, val_graphs, valid_networks

def build_dataset(case="case9", nbsamples=20, dataset_type="y_OPF", save_dataframes="./data", opf=True,
                  mutations = ["cost", "load"], mutation_rate=0.7, uniqueid=None, experiment=None,scale=True,
                  hetero=True, device="cpu"):
    print("building dataset with {nbsamples} variants")

    case_method = getattr(pp.networks, case)
    original_network = case_method()
    networks = {"original":original_network, "mutants":[]}
    network = deepcopy(original_network)
    uniqueid = uuid.uuid4() if uniqueid is None else uniqueid
    path = "."

    if save_dataframes is not None:
        graph = PandaPowerGraph(network,scale=scale, hetero=hetero)
        path = "{}/{}_{}/".format(save_dataframes,case,uniqueid)
        os.makedirs(path, exist_ok=True)
        graph.export(path+"/raw")

    transforms = [T.ToUndirected(merge=True), T.ToDevice(device)] if hetero else [T.ToDevice(device)]
    graphs = []
    sample_id= 0

    while len(graphs)<nbsamples and sample_id<nbsamples*100:
        # stop if we mutated more than 100 times the size needed without finding enough valid examples
        sample_id = sample_id+1
        print("sample id",sample_id)
        network = deepcopy(original_network)

        if mutation_rate>0:
            if "cost" in mutations:
                network = mutate_costs(network, mutation_rate=mutation_rate)
            
            if "load" in mutations:
                network = mutate_loads(network, mutation_rate=mutation_rate)

            if "load_relative" in mutations:
                network = mutate_loads(network, mutation_rate=mutation_rate, relative=True)

        try:
            if opf:
                pp.runopp(network, delta=1e-16)
            else:
                pp.runpp(network, delta=1e-16)
        except Exception as e:
            print("error in opf",e)
            continue

        networks["mutants"].append(network)
        if dataset_type=="y_no_OPF":
            graph_y = PandaPowerGraph(network,include_res=False,opf_as_y=True, preprocess='metapath2vec',
                            transform=T.Compose(transforms),scale=scale, hetero=hetero)
        elif dataset_type=="y_OPF":
            graph_y = PandaPowerGraph(network,include_res=True,opf_as_y=True, preprocess='metapath2vec',
                            transform=T.Compose(transforms),scale=scale, hetero=hetero)
        if dataset_type=="no_y_OPF":
            graph_y = PandaPowerGraph(network,include_res=True,opf_as_y=False, preprocess='metapath2vec',
                            transform=T.Compose(transforms),scale=scale, hetero=hetero)
            
        if save_dataframes is not None:
            path = "{}/{}_{}/".format(save_dataframes,case,uniqueid)
            graph_y.export(path+"op_{}".format(sample_id), experiment=experiment)

        graphs.append(graph_y)
   
    return graphs, networks, path, uniqueid


