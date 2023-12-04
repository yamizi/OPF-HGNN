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
import ray

from utils.pandapower.pandapower_graph import PandaPowerGraph
import copy
def clear_duplicates(train_graphs, train_networks, val_graphs, valid_networks):
    print("clearing duplicates")
    train_graphs_c = deepcopy(train_graphs)

    val_str = [val_graph.data.to_dict().__str__() for val_graph in val_graphs]
    train_str = [train_graph.data.to_dict().__str__() for train_graph in train_graphs_c]

    comparisons = np.array([a==b for (a,b) in itertools.product(val_str, train_str)]).reshape(len(val_str),len(train_str))

    nb_duplicates = np.sum(comparisons)
    print("Found ",nb_duplicates," duplicates")

    return train_graphs, train_networks, val_graphs, valid_networks

@ray.remote
def build_one_graph_ray(sample_id, original_network, mutations,mutation_rate,opf,transforms, scale, dataset_type, hetero,
                    save_dataframes,case, uniqueid, experiment, device="cpu" ):
    return build_one_graph(sample_id, original_network, mutations,mutation_rate,opf,transforms, scale, dataset_type, hetero,
                    save_dataframes,case, uniqueid, experiment, device )
def build_one_graph(sample_id, original_network, mutations,mutation_rate,opf,transforms, scale, dataset_type, hetero,
                    save_dataframes,case, uniqueid, experiment, device="cpu" ):
    network = deepcopy(original_network)

    if mutation_rate>0:
        if "cost" in mutations:
            network = mutate_costs(network, mutation_rate=mutation_rate)
        
        if "load" in mutations:
            network = mutate_loads(network, mutation_rate=mutation_rate)

        if "load_relative" in mutations:
            network = mutate_loads(network, mutation_rate=mutation_rate, relative=True)

    try:
        run_errors = pp.diagnostic(copy.deepcopy(network), report_style="compact")
        network.original_errors = run_errors
        print(run_errors)
        if opf==2:
            pp.runpm_ac_opf(network)
        elif opf==1:
            pp.runopp(network)
        else:
            pp.runpp(network)
    except Exception as e:
        print("error in opf",e)
        return None, None

    
    if dataset_type=="y_no_OPF":
        graph_y = PandaPowerGraph(network,include_res=False,opf_as_y=True, preprocess='metapath2vec',
                        transform=T.Compose(transforms),scale=scale, hetero=hetero, device=device)
    elif dataset_type=="y_OPF":
        graph_y = PandaPowerGraph(network,include_res=True,opf_as_y=True, preprocess='metapath2vec',
                        transform=T.Compose(transforms),scale=scale, hetero=hetero, device=device)
    if dataset_type=="no_y_OPF":
        graph_y = PandaPowerGraph(network,include_res=True,opf_as_y=False, preprocess='metapath2vec',
                        transform=T.Compose(transforms),scale=scale, hetero=hetero, device=device)
        
    if save_dataframes is not None:
        path = "{}/{}_{}/".format(save_dataframes,case,uniqueid)
        graph_y.export(path+"op_{}".format(sample_id), experiment=experiment)

    return graph_y, network

def build_dataset(case="case9", nbsamples=20, dataset_type="y_OPF", save_dataframes="./data", opf=1,
                  mutations = ["cost", "load"], mutation_rate=0.7, uniqueid=None, experiment=None,scale=True,
                  hetero=True, device="cpu",use_ray=True):
    print(f"building dataset with {nbsamples} variants, ray {use_ray} and device {device}")

    case_method = getattr(pp.networks, case)
    original_network = case_method()
    networks = {"original":original_network, "mutants":[]}
    network = deepcopy(original_network)
    uniqueid = uuid.uuid4() if uniqueid is None else uniqueid
    path = "."

    if save_dataframes is not None:
        graph = PandaPowerGraph(network,scale=scale, hetero=hetero, device=device)
        path = "{}/{}_{}/".format(save_dataframes,case,uniqueid)
        os.makedirs(path, exist_ok=True)
        graph.export(path+"/raw")

    transforms = [T.ToUndirected(merge=True), T.ToDevice(device)] if hetero else [T.ToDevice(device)]
    transforms = [T.ToDevice(device)] +transforms
    graphs = []
    sample_id= 0

    
    
    while len(graphs)<nbsamples and sample_id<nbsamples*100:
        # stop if we mutated more than 100 times the size needed without finding enough valid examples
        print("sample id",sample_id)

        if use_ray:
            graph_y_network = [build_one_graph_ray.remote(sample_id, original_network, mutations,mutation_rate,opf,transforms, scale, dataset_type, hetero,
                        save_dataframes,case, uniqueid, experiment=None, device=device) for sample_id in range(nbsamples)]

            graph_y_network = ray.get(graph_y_network)
        else:
            graph_y_network = [build_one_graph(sample_id, original_network, mutations,mutation_rate,opf,transforms, scale, dataset_type, hetero,
                        save_dataframes,case, uniqueid, experiment=experiment, device=device) for sample_id in range(nbsamples)]

        graph_y_networks = [g for g in graph_y_network if g[0] is not None]
        graph_y, networks_y = list(zip(*graph_y_networks))

        graphs = graph_y+ graph_y
        networks["mutants"] =  networks["mutants"] + list(networks_y)
   
    networks["mutants"] = networks["mutants"][:nbsamples]
    graphs = graphs[:nbsamples]
    return graphs, networks, path, uniqueid


