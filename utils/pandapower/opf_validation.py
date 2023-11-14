import torch
import pandapower as pp
import numpy as np
from itertools import chain
import ray


@ray.remote
def is_network_valid(i, network, y_nodes, output_nodes, nb_gens):
    valid_min_max = True
    boundaries = {}
    for node in y_nodes:
        values = output_nodes.get(node)
        if len(values)==0:
            continue
        valid_max = values[i*nb_gens[node]:(i+1)*nb_gens[node]].numpy() < getattr(network,node)[["max_p_mw","max_q_mvar"]].values
        getattr(network,node)[["max_p_mw","max_q_mvar"]] = values[i*nb_gens[node]:(i+1)*nb_gens[node]]
        
        valid_min = getattr(network,node)[["min_p_mw","min_q_mvar"]].values < values[i*nb_gens[node]:(i+1)*nb_gens[node]].numpy()
        getattr(network,node)[["min_p_mw","min_q_mvar"]] = values[i*nb_gens[node]:(i+1)*nb_gens[node]]

        print(node,": Valid min values respected:", valid_min.all(), "Valid max values respected:", valid_max.all())
        valid_min_max = valid_min_max & valid_max.all() & valid_min.all()

        boundaries = {**boundaries, node+"_min":not valid_min.all(), node+"_max":not valid_max.all()}



    run_valid = True
    run_errors = pp.diagnostic(network, report_style="compact")
    if run_errors != {} and list(run_errors.keys())!=['impedance_values_close_to_zero']:
        run_valid = False
        print(run_errors)
    
    valid = run_valid & valid_min_max
    return int(valid), {"run":run_errors, **boundaries}

def validate_opf(networks, val_graphs, outputs, y_nodes, hetero=True):
    (out_all, val_losses_all) = outputs
    if hetero:
        output_nodes = {node:torch.cat([e[node] for e in out_all],0) for node in y_nodes}
        nb_gens = {node:len(networks.get("original")[node]) for node in y_nodes}
    else:
        bus_nodes = torch.cat([e for e in out_all],0)
        y_nodes = ["ext_grid","gen","sgen"]
        nb_gens = {node:len(networks.get("original")[node]) for node in y_nodes}
        mask = list(chain.from_iterable([[e]*k for (e,k) in nb_gens.items()]))*len(networks.get("mutants"))
        output_nodes = {node:bus_nodes[np.array(mask)==node,:] for node in y_nodes}
    valid_networks = []
    errors = []

    validation = [is_network_valid.remote(i, network, y_nodes, output_nodes, nb_gens)  for i, network in enumerate(networks.get("mutants"))]
    validation_list = ray.get(validation)
    valids, errors = list(zip(*validation_list))
    valid_networks = [validation_list[i] for i,valid in enumerate(valids) if valids]

    print("OPF validation over, nb_valid:",np.mean(valids))
    return valid_networks, errors
    