from utils.appconfig import COMET_APIKEY
from comet_ml import Experiment
import time
import torch
from itertools import chain
import numpy as np
import pandas as pd
import json

class NumpyEncoder(json.JSONEncoder):
    import numpy as np
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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


def log_dataframe(dic,experiment, name, base=0, limit=0):
    initial_val = len(list(dic.values())[0])
    equal_nb = all([initial_val == len(val) for val in dic.values()])
    if equal_nb:
        experiment.log_dataframe_profile(pd.DataFrame(dic),name=f"{name}_{base}")
    else:
        for (e, v) in dic.items():
            experiment.log_dataframe_profile(pd.DataFrame(v), name=f"{name}_{base}_{e}")


def log_dict_series(dic, experiment, limit=0):
    counter = 0
    for (e, v) in dic.items():
        for i, l in enumerate(v):
            if limit>0 and i > limit:
                break
            experiment.log_metric(e, l, step=i)
            counter+=1

        if counter>9000:
            time.sleep(60)
            counter = 0


def log_opf(networks, val_graphs, outputs, y_nodes, experiment, hetero=True):
    (out_all, val_losses_all) = outputs
    if hetero:
        output_nodes = {node: torch.cat([e[node] for e in out_all], 0) for node in y_nodes}
        labels = {node: torch.cat([e.data[node].y for e in val_graphs]) for node in y_nodes}
    else:
        bus_nodes = torch.cat([e for e in out_all], 0)
        y_nodes = ["ext_grid", "gen", "sgen"]
        nb_gens = {node: len(networks.get("original")[node]) for node in y_nodes}
        mask = list(chain.from_iterable([[e] * k for (e, k) in nb_gens.items()])) * len(networks.get("mutants"))
        output_nodes = {node: bus_nodes[np.array(mask) == node, :] for node in y_nodes}

        mask_nan = ~torch.isnan(val_graphs[0].data.y).any(1)
        ground_truth = torch.cat([e.data.y[mask_nan] for e in val_graphs])
        labels = {node: ground_truth[np.array(mask) == node, :] for node in y_nodes}

    delta = 1e-10  # to avoid division by zero
    dic = {}
    for node, outputs in output_nodes.items():
        ground_truth = labels[node]

        if node in ["gen","ext_grid"]:
            dic = {**dic,
                   "P_pred_" + node: outputs[:, 0].cpu().numpy(), "P_true_" + node: ground_truth[:, 0].cpu().numpy(),
                   "Q_pred_" + node: outputs[:, 1].cpu().numpy(), "Q_true_" + node: ground_truth[:, 1].cpu().numpy()}

            SE_P = (outputs[:, 0].cpu().numpy() - ground_truth[:, 0].cpu().numpy()) ** 2
            relativeSE_P = SE_P / (ground_truth[:, 0].cpu().numpy() ** 2 + delta)

            SE_Q = (outputs[:, 1].cpu().numpy() - ground_truth[:, 1].cpu().numpy()) ** 2
            relativeSE_Q = SE_P / (ground_truth[:, 1].cpu().numpy() ** 2 + delta)

            dic = {**dic, "SE_P_" + node: SE_P, "relativeSE_P_" + node: relativeSE_P, "SE_Q_" + node: SE_Q,
                   "relativeSE_Q_" + node: relativeSE_Q}
        if node in ["bus"]:
            dic = {"Vm_pred_" + node: outputs[:, 2].cpu().numpy(), "Vm_true_" + node: ground_truth[:, 0].cpu().numpy(),
                   "Va_pred_" + node: outputs[:, 3].cpu().numpy(), "Va_true_" + node: ground_truth[:, 1].cpu().numpy(),
                   **dic}

            SE_Vm = (outputs[:, 2].cpu().numpy() - ground_truth[:, 0].cpu().numpy()) ** 2
            relativeSE_Vm = SE_Vm / (ground_truth[:, 0].cpu().numpy() ** 2 + delta)

            SE_Va = (outputs[:, 3].cpu().numpy() - ground_truth[:, 1].cpu().numpy()) ** 2
            relativeSE_Va = SE_Va / (ground_truth[:, 1].cpu().numpy() ** 2 + delta)

            dic = {**dic, "SE_Vm_" + node: SE_Vm, "relativeSE_Vm_" + node: relativeSE_Vm, "SE_Va_" + node: SE_Va,
                   "relativeSE_Va_" + node: relativeSE_Va}

        if node in ["line"]:
            dic = {"pl_mw_pred_" + node: outputs[:, 4].cpu().numpy(), "pl_mw_true_" + node: ground_truth[:, 0].cpu().numpy(),
                   "Va_pred_" + node: outputs[:, 5].cpu().numpy(), "Va_true_" + node: ground_truth[:, 1].cpu().numpy(),
                   **dic}

            SE_pl_mw = (outputs[:, 4].cpu().numpy() - ground_truth[:, 0].cpu().numpy()) ** 2
            relativeSE_pl_mw = SE_pl_mw / (ground_truth[:, 0].cpu().numpy() ** 2 + delta)

            SE_ql_mvar = (outputs[:, 5].cpu().numpy() - ground_truth[:, 1].cpu().numpy()) ** 2
            relativeSE_ql_mvar = SE_ql_mvar / (ground_truth[:, 1].cpu().numpy() ** 2 + delta)

            dic = {**dic, "SE_pl_mw_" + node: SE_pl_mw, "relativeSE_pl_mw_" + node: relativeSE_pl_mw, "SE_ql_mvar_" + node: SE_ql_mvar,
                   "relativeSE_ql_mvar_" + node: relativeSE_ql_mvar}

            dic = {"i_from_ka_pred_" + node: outputs[:, 6].cpu().numpy(),
                   "i_from_ka_true_" + node: ground_truth[:, 2].cpu().numpy(),
                   "Va_pred_" + node: outputs[:, 7].cpu().numpy(), "Va_true_" + node: ground_truth[:, 3].cpu().numpy(),
                   **dic}

            SE_i_from_ka = (outputs[:, 6].cpu().numpy() - ground_truth[:, 2].cpu().numpy()) ** 2
            relativeSE_i_from_ka = SE_i_from_ka / (ground_truth[:, 2].cpu().numpy() ** 2 + delta)

            SE_i_to_ka = (outputs[:, 7].cpu().numpy() - ground_truth[:, 3].cpu().numpy()) ** 2
            relativeSE_i_to_ka = SE_i_to_ka / (ground_truth[:, 3].cpu().numpy() ** 2 + delta)

            dic = {**dic, "SE_i_to_ka_" + node: SE_i_to_ka, "relativeSE_i_to_ka_" + node: relativeSE_i_to_ka,
                   "SE_i_from_ka_" + node: SE_i_from_ka,
                   "relativeSE_i_from_ka_" + node: relativeSE_i_from_ka}

    if experiment:
        log_dict_series(dic, experiment)

    return dic
