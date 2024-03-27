from typing import Callable, Optional
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset
)
import pandapower as pp

import numpy as np

from copy import deepcopy
import psutil, gc
import torch_geometric.transforms as T
import json
import uuid
import os
from utils.pandapower.mutations import mutate_costs, mutate_loads
import itertools
import ray
import time
from utils.pandapower.pandapower_graph import PandaPowerGraph
from runs.matpower import opf as matpower_opf
import copy


def clear_duplicates(train_graphs, train_networks, val_graphs, valid_networks):
    print("clearing duplicates")
    train_graphs_c = deepcopy(train_graphs)

    val_str = [val_graph.data.to_dict().__str__() for val_graph in val_graphs]
    train_str = [train_graph.data.to_dict().__str__() for train_graph in train_graphs_c]

    comparisons = np.array([a==b for (a,b) in itertools.product(val_str, train_str)]).reshape(len(val_str),len(train_str))

    nb_duplicates = np.sum(comparisons)
    print("Found ",nb_duplicates," duplicates")

    correct = np.where(comparisons.sum(0)==0)[0]
    train_graphs_filtered = [train_graphs[i] for i in correct]
    train_mutants_filtered = [train_networks.get("mutants")[i] for i in correct]
    train_convergence_filtered = [train_networks.get("convergence_time")[i] for i in correct]

    train_networks_filtered = {"mutants":train_mutants_filtered, "original":train_networks.get("original"),
                               "convergence_time":train_convergence_filtered}
    return train_graphs_filtered, train_networks_filtered, val_graphs, valid_networks


def build_batch_graph(sample_id, network, mutations,mutation_rate,opf,transforms, scale, dataset_type, hetero,
                    save_dataframes,case, uniqueid, experiment, batch_size=25, device="cpu"):
    assert opf==3 and "load_relative" in mutations  and "OPF" in dataset_type, "Batch graph generation is only available with matpower and load relative"

    graphs = []
    loads = [mutate_loads(copy.deepcopy(network), mutation_rate=mutation_rate, relative=True)[1] for i in range(batch_size)]
    octave_path = os.environ.get("OCTAVE_PATH", None)
    #try:
    if True:
        networks, convergence_times = matpower_opf(case=case, all_loads=loads, octave_path=octave_path, batch_size=batch_size)
        for i, network in enumerate(networks):
            if network is None:
                continue

            graph_y = PandaPowerGraph(network, include_res=True, opf_as_y=True, preprocess='metapath2vec',
                                      transform=T.Compose(transforms), scale=scale, hetero=hetero, device=device)

            graphs.append((graph_y, network, convergence_times[i]))

    # try:
    #     pass
    # except Exception as e:
    #     print("opf 3 error", e)
    #     return graphs

    return graphs

@ray.remote
def build_one_graph_ray(sample_id, original_network, mutations,mutation_rate,opf,transforms, scale, dataset_type, hetero,
                    save_dataframes,case, uniqueid, experiment, device="cpu" ):
    return build_one_graph(sample_id, original_network, mutations,mutation_rate,opf,transforms, scale, dataset_type, hetero,
                    save_dataframes,case, uniqueid, experiment, device )
def build_one_graph(sample_id, network, mutations,mutation_rate,opf,transforms, scale, dataset_type, hetero,
                    save_dataframes,case, uniqueid, experiment, device="cpu" ):
    gc.collect()
    convergence_time = 0
    if mutation_rate>0:
        if "cost" in mutations:
            network, masked = mutate_costs(copy.deepcopy(network), mutation_rate=mutation_rate)
        
        if "load" in mutations:
            network, loads = mutate_loads(copy.deepcopy(network), mutation_rate=mutation_rate)

        if "load_relative" in mutations:
            network, loads = mutate_loads(copy.deepcopy(network), mutation_rate=mutation_rate, relative=True)

    if opf==3:
        octave_path = os.environ.get("OCTAVE_PATH",None)
        try:
            network, convergence_time = matpower_opf(case=case,loads=loads,octave_path=octave_path)
            network = network[0]
            convergence_time = convergence_time[0]
            if network is None:
                return None,None,None
        except Exception as e:
            print("opf 3 error", e)
            return None, None, None
    else:
        #fix minimum r_ohm and clean diagnostic warning
        network.line.r_ohm_per_km = network.line.r_ohm_per_km.clip(0.011)

        try:
            run_errors = pp.diagnostic(copy.deepcopy(network), report_style="compact")
            network.original_errors = run_errors
            print(run_errors)
            init = time.time()
            if opf==2:
                pp.runpm_ac_opf(network)
            elif opf==1:
                pp.runopp(network)
            else:
                pp.runpp(network)
            convergence_time = time.time()-init

            if not network.OPF_converged:
                print("not converged opf")
                return None, None, None
        except Exception as e:
            print("error in opf",e)
            return None, None, None

    if dataset_type=="y_no_OPF":
        graph_y = PandaPowerGraph(network,include_res=False,opf_as_y=True, preprocess='metapath2vec',
                        transform=T.Compose(transforms),scale=scale, hetero=hetero, device=device)
    elif dataset_type=="y_OPF":
        graph_y = PandaPowerGraph(network,include_res=True,opf_as_y=True, preprocess='metapath2vec',
                        transform=T.Compose(transforms),scale=scale, hetero=hetero, device=device)
    elif dataset_type=="no_y_OPF":
        graph_y = PandaPowerGraph(network,include_res=True,opf_as_y=False, preprocess='metapath2vec',
                        transform=T.Compose(transforms),scale=scale, hetero=hetero, device=device)
        
    if save_dataframes is not None:
        path = "{}/{}_{}/".format(save_dataframes,case,uniqueid)
        graph_y.export(path+"op_{}".format(sample_id), experiment=experiment)

    return graph_y, network, convergence_time

def build_dataset(case="case9", nbsamples=20, dataset_type="y_OPF", save_dataframes="./data", opf=1,
                  mutations = ["cost", "load"], mutation_rate=0.7, uniqueid=None, experiment=None,scale=True,
                  hetero=True, device="cpu",use_ray=True, batch_size=100):
    print(f"building dataset with {nbsamples} variants, ray {use_ray} and device {device}")

    case_method = getattr(pp.networks, case)
    original_network = case_method()
    networks = {"original":original_network, "mutants":[], "convergence_time":[]}
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
    if scale:
        transforms.append((T.NormalizeFeatures()))
        scale = False

    graphs = []
    sample_id= 0

    while len(graphs)<nbsamples and sample_id<nbsamples*100:
        # stop if we mutated more than 100 times the size needed without finding enough valid examples
        print("loop sample id",sample_id," total graphs",len(graphs))
        sample_id = sample_id+nbsamples

        if use_ray:
            graph_y_network = [build_one_graph_ray.remote(sample_id, original_network, mutations,mutation_rate,opf,transforms, scale, dataset_type, hetero,
                        save_dataframes,case, uniqueid, experiment=None) for sample_id in range(nbsamples)]

            graph_y_network = ray.get(graph_y_network)
        else:

            if opf==3 and batch_size>1:
                graph_y_network = []
                for i in range(len(graphs),nbsamples,batch_size):
                    print("batch generation ",i,"+",batch_size,"/",nbsamples)
                    step_network = build_batch_graph(sample_id, original_network, mutations,mutation_rate,opf,transforms, scale, dataset_type, hetero,
                        save_dataframes,case, uniqueid, experiment=experiment, batch_size=batch_size)
                    graph_y_network = graph_y_network + step_network
                    gc.collect()
            else:
                graph_y_network = [build_one_graph(sample_id, original_network, mutations,mutation_rate,opf,transforms, scale, dataset_type, hetero,
                            save_dataframes,case, uniqueid, experiment=experiment) for sample_id in range(nbsamples)]

        graph_y_networks = [g for g in graph_y_network if g[0] is not None]
        graph_y, networks_y, convergence_times = list(zip(*graph_y_networks)) if len(graph_y_networks) else ([],[], [])

        graphs = graphs+ list(graph_y)
        networks["mutants"] =  networks["mutants"] + list(networks_y)
        networks["convergence_time"] = networks["convergence_time"] + list(convergence_times)
   
    networks["mutants"] = networks["mutants"][:nbsamples]
    networks["convergence_time"] = networks["convergence_time"][:nbsamples]
    graphs = graphs[:nbsamples]
    return graphs, networks, path, uniqueid


