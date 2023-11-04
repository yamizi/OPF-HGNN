from utils.appconfig import COMET_APIKEY
from comet_ml import Experiment
import time
import torch

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


def log_opf(val_graphs, outputs, y_nodes, experiment):
    (out_all, val_losses_all) = outputs
    output_nodes = {node:torch.cat([e[node] for e in out_all],0) for node in y_nodes}

    for node, outputs in output_nodes.items():
        ground_truth = torch.cat([e.data[node].y for e in val_graphs])

        dic = {"P_pred_"+node:outputs[:,0].cpu().numpy(), "P_true_"+node:ground_truth[:,0].cpu().numpy(),
               "Q_pred_"+node:outputs[:,1].cpu().numpy(), "Q_true_"+node:ground_truth[:,1].cpu().numpy()}
        log_dict_series(dic,experiment)