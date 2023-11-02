import torch
import pandapower as pp

def is_network_valid(i, network, y_nodes, output_nodes, nb_gens):
    valid_min_max = True
    for node in y_nodes:
        values = output_nodes.get(node)
        valid_max = values[i*nb_gens[node]:(i+1)*nb_gens[node]].numpy() < getattr(network,node)[["max_p_mw","max_q_mvar"]].values
        getattr(network,node)[["max_p_mw","max_q_mvar"]] = values[i*nb_gens[node]:(i+1)*nb_gens[node]]
        
        valid_min = getattr(network,node)[["min_p_mw","min_q_mvar"]].values < values[i*nb_gens[node]:(i+1)*nb_gens[node]].numpy()
        getattr(network,node)[["min_p_mw","min_q_mvar"]] = values[i*nb_gens[node]:(i+1)*nb_gens[node]]

        print(node,": Valid min values respected:", valid_min.all(), "Valid max values respected:", valid_max.all())
        valid_min_max = valid_min_max & valid_max.all() & valid_min.all()

    run_errors = pp.diagnostic(network, report_style="compact")
    if run_errors != {}:
        print(run_errors)
    
    valid = run_errors == {} & valid_min_max
    return valid

def validate_opf(networks, val_graphs, outputs, y_nodes):
    (out_all, val_losses_all) = outputs
    output_nodes = {node:torch.cat([e[node] for e in out_all],0) for node in y_nodes}
    nb_gens = {node:len(networks.get("original")[node]) for node in y_nodes}
    valid_networks = []
    for i, network in enumerate(networks.get("mutants")):
        valid =  is_network_valid(i, network, y_nodes, output_nodes, nb_gens)
        if valid:
            valid_networks.append(network)

    return valid_networks
    