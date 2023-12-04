import sys

sys.path.append(".")
import uuid
from matplotlib import pyplot as plt

from utils.logging import init_comet, log_dict_series, log_opf
from utils.pandapower import build_dataset, clear_duplicates
from utils.pandapower.opf_validation import validate_opf
from torch_geometric.nn import to_hetero

from torch_geometric.loader import DataLoader
from utils.models import GNN
import torch
from utils.train import train_opf
from utils.plot import plot_losses, plot_results
import json


def run_case(training_cases=[["case9", 64, 0.7, ["cost", "load"]]], experiment=None,
             validation_case=["case9", 64, 0.7, ["cost", "load"]], plot=True,
             save_path="./output", title="", dataset_type="y_OPF", scale=False,
             max_epochs=500, y_nodes=["gen", "ext_grid","bus"], train_batch_size=5, val_batch_size=5,
             device="cuda", filter=True, opf=2,use_ray=True):
    uniqueid = uuid.uuid4()
    if experiment is not None:
        experiment.log_parameters({"uniqueid": uniqueid, "max_epochs": max_epochs, "dataset_type": dataset_type,
                                   "scale": scale, "type": "hetero", "opf": opf,"use_ray":use_ray,
                                   "save_path": save_path, "title": title, "y_nodes": y_nodes, "plot": plot,
                                   "device": device})

    if torch.cuda.is_available() and "cuda" in device:
        device = device
        print("running with ", device)
    else:
        device = "cpu"
        print("running with cpu backend")

    common_params = {"dataset_type": dataset_type, "save_dataframes": save_path, "experiment": experiment,
                     "scale": scale, "device": device, "opf": opf,"use_ray":use_ray}

    val_case_name, nb_graphs, mutation_rate, mutations = validation_case
    val_graphs, valid_networks, _, _ = build_dataset(val_case_name, nbsamples=nb_graphs,
                                                     mutation_rate=mutation_rate, uniqueid="{}/val".format(uniqueid),
                                                     mutations=mutations, **common_params)
    print("Correct validation graphs {}/{}".format(len(val_graphs), nb_graphs))
    experiment.log_metric("nb_valid_graphs", len(val_graphs))

    train_graphs = []
    train_networks = {"mutants": []}
    nb_graphs = 0
    train_case_name = ""
    for training_case in training_cases:
        train_case_name, nb_graph, mutation_rate, mutations = training_case

        train_graph, train_network, _, _ = build_dataset(train_case_name, nbsamples=nb_graph,
                                                         mutation_rate=mutation_rate,
                                                         uniqueid="{}/train".format(uniqueid),
                                                         mutations=mutations, **common_params)

        train_graphs += train_graph
        train_networks["mutants"] += train_network["mutants"]
        nb_graphs += nb_graph

    if filter:
        train_graphs, train_networks, val_graphs, valid_networks = clear_duplicates(train_graphs, train_networks,
                                                                                    val_graphs, valid_networks)

    print("Correct training graphs {}/{}".format(len(train_graphs), nb_graphs))
    experiment.log_metric("nb_train_graphs", len(train_graphs))

    train_loader = DataLoader([g[0] for g in train_graphs], batch_size=train_batch_size)
    val_loader = DataLoader([g[0] for g in val_graphs], batch_size=val_batch_size)

    if len(train_graphs) == 0:
        return

    graph_y = train_graphs[0]
    data = graph_y[0]
    model = GNN(hidden_channels=64, out_channels=graph_y.num_outputs)
    model = to_hetero(model, data.metadata(), aggr='sum').to(device)

    print("model device",next(model.parameters()).device)

    train_losses, val_losses, val_losses_nodes, last_out, b_train_losses, b_val_losses, lr = train_opf(
        model, train_loader, val_loader, max_epochs=max_epochs, y_nodes=y_nodes, device=device)

    val_losses_gen, val_losses_ext_grid, val_losses_bus = val_losses_nodes
    constrained_networks, errors_network = validate_opf(valid_networks, val_graphs, last_out, y_nodes=y_nodes, opf=opf,
                                                        use_ray=use_ray)

    case_name = "{}->{}".format(train_case_name, val_case_name)

    log_dict = {"constraint_boundary": constrained_networks[:, 1].tolist(),
                "constraint_opf": constrained_networks[:, 0].tolist(),
                "constraint": constrained_networks.prod(1).tolist(), "train_losses": train_losses,
                "val_losses": val_losses,
                "b_train_losses": b_train_losses, "b_val_losses": b_val_losses, "learning_rate": lr,
                "val_losses_gen": val_losses_gen, "val_losses_ext_grid": val_losses_ext_grid,
                "val_losses_bus":val_losses_bus}

    with open(save_path + "/losses.json", "w") as outfile:
        json.dump(log_dict, outfile)

    if experiment is not None:
        log_dict_series(log_dict, experiment)
        log_opf(valid_networks, val_graphs, last_out, y_nodes, experiment)

    if plot:
        plot_losses(train_losses, val_losses, val_losses_gen, val_losses_ext_grid, case_name, title, save_path)
        plot_results(valid_networks, val_graphs, last_out, y_nodes, constrained_networks, errors_network, case_name,
                     title, save_path)


if __name__ == "__main__":
    max_epochs = 5
    case = "case14"
    mutation = "load_relative"
    training_case = [[case, 8, 0.7, [mutation]]]
    validation_case = [case, 2, 0.7, [mutation]]
    opf = 1

    experiment = init_comet({"case": case, "mutation": mutation})
    run_case(training_cases=training_case, validation_case=validation_case, val_batch_size=50, train_batch_size=32,
             title="generalization load_relative", save_path=f"./output/case{training_case}_{validation_case}",
             max_epochs=max_epochs, experiment=experiment, dataset_type="y_no_OPF",
             scale=False, filter=True, opf=opf,use_ray=False)
    plt.show()
    exit()

    training_case = [["case9", 64, 0.7, ["cost"]]]
    validation_case = ["case14", 32, 0.7, ["cost"]]
    run_case(training_cases=training_case, validation_case=validation_case,
             title="generalization cost", save_path="./output/case9_14")
