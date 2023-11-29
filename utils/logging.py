from utils.appconfig import COMET_APIKEY
from comet_ml import Experiment
import time
import torch
from itertools import chain
import numpy as np

def init_comet(args, project_name="debug", workspace="hgnn"):
    timestamp = time.time()
    args["timestamp"] = timestamp
    experiment = Experiment(api_key=COMET_APIKEY,
                            project_name=project_name,
                            workspace=workspace,
                            auto_param_logging=False, auto_metric_logging=False,
                            parse_args=False, display_summary=False, disabled=False)
    experiment.log_parameters(args)

    return experiment


def log_dict_series(dic, experiment):
    [[experiment.log_metric(e, l, step=i) for i, l in enumerate(v)] for (e,v) in dic.items()]


def log_opf(networks,val_graphs, outputs, y_nodes, experiment, hetero=True):
    (out_all, val_losses_all) = outputs
    if hetero:
        output_nodes = {node:torch.cat([e[node] for e in out_all],0) for node in y_nodes}
        labels = {node: torch.cat([e.data[node].y for e in val_graphs]) for node in y_nodes}
    else:
        bus_nodes = torch.cat([e for e in out_all],0)
        y_nodes = ["ext_grid","gen","sgen"]
        nb_gens = {node:len(networks.get("original")[node]) for node in y_nodes}
        mask = list(chain.from_iterable([[e]*k for (e,k) in nb_gens.items()]))*len(networks.get("mutants"))
        output_nodes = {node:bus_nodes[np.array(mask)==node,:] for node in y_nodes}

        mask_nan =~torch.isnan(val_graphs[0].data.y).any(1)
        ground_truth = torch.cat([e.data.y[mask_nan] for e in val_graphs])
        labels = {node:ground_truth[np.array(mask)==node,:] for node in y_nodes} 

    for node, outputs in output_nodes.items():
        ground_truth = labels[node]
        dic = {"P_pred_"+node:outputs[:,0].cpu().numpy(), "P_true_"+node:ground_truth[:,0].cpu().numpy(),
               "Q_pred_"+node:outputs[:,1].cpu().numpy(), "Q_true_"+node:ground_truth[:,1].cpu().numpy()}

        if node in ["bus","gen","sgen"]:
            dic = {"Vm_pred_" + node: outputs[:, 2].cpu().numpy(), "Vm_true_" + node: ground_truth[:, 2].cpu().numpy(),
                   "Va_pred_" + node: outputs[:, 3].cpu().numpy(), "Va_true_" + node: ground_truth[:, 3].cpu().numpy(),
                   **dic}
        log_dict_series(dic,experiment)