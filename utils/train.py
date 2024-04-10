import torch.nn.functional as F
import torch
import numpy as np
from utils.models import FCNN, GNN
from torch_geometric.nn import to_hetero
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from torch_geometric.loader import DataLoader
import pickle
import psutil, gc
from utils.losses import boundary_loss, node_loss, relative_loss, power_imbalance_loss


def clamp_boundaries(boundaries, y, node=None, mask=None):
    if mask is not None:
        masked_y = y[mask.bool()].reshape((y.shape[0], -1))
    else:
        masked_y = y

    min_boundaries = torch.stack([boundaries[:, 2 * i] for i in range(y.shape[1])
                                  if torch.isnan(boundaries[:, 2 * i]).sum() == 0], 1)

    max_boundaries = torch.stack([boundaries[:, 2 * i + 1] for i in range(y.shape[1])
                                  if torch.isnan(boundaries[:, 2 * i + 1]).sum() == 0], 1)

    masked_y = torch.clamp(masked_y, min_boundaries, max_boundaries)

    clamped = y * (1 - mask)
    clamped[mask.bool()] = masked_y.flatten()
    return clamped


def auto_garbage_collect(pct=50.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return


def objective(config, graph, device, max_epochs, y_nodes, hetero, train_loader, val_loader, clamp_boundary,
              use_physical_loss=1):
    # print("objective", config)  # ①

    model = GNN(hidden_channels=[config.get("hidden_channels") for i in range(config.get("nb_hidden_layers"))],
                out_channels=graph.num_outputs, aggr=config.get("aggr"), cls=config.get("cls"))

    metadata = graph[0].cpu().metadata()
    model = to_hetero(model, metadata, aggr='sum')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr"))
    milestones = [max_epochs // 2, (max_epochs * 3) // 4, (max_epochs * 9) // 10]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones,
                                                        gamma=config.get("decay_lr"))
    neighboorhood = None
    while True:
        for batch in train_loader:
            out, labels, loss, losses, b_losses, neighboorhood = train_step(model, optimizer, batch, None, y_nodes, node_loss, hetero,
                                                     clamp_boundary=(clamp_boundary == 1 or clamp_boundary == 2),
                                                     use_physical_loss=use_physical_loss, neighboorhood=neighboorhood)
        lr_scheduler.step()

        val_loss_gen = 0
        val_loss_ext_grid = 0
        val_loss_bus = 0
        val_loss_line = 0

        for batch in val_loader:
            last_out, last_label, loss, losses, b_losses, p_losses, neighboorhood = eval_step(model, batch, None,
                  y_nodes, node_loss, hetero, clamp_boundary=(clamp_boundary == 2 or clamp_boundary == 3),
                   neighboorhood=neighboorhood, use_physical_loss=use_physical_loss)

            val_loss_gen += losses[0].mean()
            val_loss_ext_grid += losses[1].mean() if len(losses) > 1 else 0
            val_loss_bus += losses[2].mean() if len(losses) > 2 else 0
            val_loss_line += losses[3].mean() if len(losses) > 3 else 0

        auto_garbage_collect()

        ray.train.report({"val_loss_gen": val_loss_gen, "val_loss_ext_grid": val_loss_ext_grid,
                          "val_loss_bus": val_loss_bus, "val_loss_line": val_loss_line})  # Report to Tune


def train_cv(pickle_file, cv_ratio, graph, max_epochs=20, num_samples=10, y_nodes=["gen", "ext_grid", "bus"],
             device="cpu", hetero=True, base_lr=[0.1, 1], plot=False, num_graphs=1000, train_batch_size=32,
             val_batch_size=32, train_graphs=None, clamp_boundary=0, use_physical_loss=1):
    if train_graphs is None:
        with open(pickle_file, "rb") as pickled:
            print("Loading cross validation dataset from", pickle_file)
            loaded = pickle.load(pickled)

        train_graphs = loaded.get("train_graphs")
    train_list = [train_graphs[i][0].to(device) for i in range(num_graphs)]

    del train_graphs

    cv_list = train_list[:min(len(train_list), 1000)]
    nb_train = int(cv_ratio * len(cv_list))
    cv_train = cv_list[nb_train:]
    training_loader = DataLoader(cv_train, batch_size=val_batch_size)
    cv_val = cv_list[:nb_train]
    validation_loader = DataLoader(cv_val, batch_size=train_batch_size)

    lr_space = np.logspace(np.round(np.log(base_lr[0]) / np.log(10)), np.round(np.log(base_lr[1]) / np.log(10)), 3, 10)
    lr_decay = np.linspace(0.1, 0.9, 5)

    search_space = {"lr": tune.choice(lr_space.tolist()), "decay_lr": tune.choice(lr_decay.tolist()),
                    "hidden_channels": tune.choice([32, 64, 128, 256]), "nb_hidden_layers": tune.choice([1, 2, 3, 4]),
                    "aggr": tune.choice(["mean", "max"]), "cls": tune.choice(["gcn", "sage", "gat"])}

    print("running optuna search on ", search_space, "for epochs", max_epochs, "and size", len(training_loader.dataset)
          , "using device", device)

    metrics = ["val_loss_bus", "val_loss_gen", "val_loss_ext_grid"]
    # metrics = ["val_loss_gen", "val_loss_ext_grid"]
    algo = OptunaSearch(metric=metrics, mode=["min"] * len(metrics))  # ②

    # ray.init(num_cpus=10)

    tuner = tune.Tuner(  # ③
        tune.with_parameters(objective, graph=ray.put(graph), device=device, max_epochs=max_epochs, y_nodes=y_nodes,
                             hetero=hetero, train_loader=ray.put(training_loader),
                             val_loader=ray.put(validation_loader), clamp_boundary=clamp_boundary,
                            use_physical_loss=use_physical_loss),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=num_samples,
        ),
        run_config=ray.train.RunConfig(
            stop={"training_iteration": max_epochs},
        ),
        param_space=search_space
    )
    results = tuner.fit()
    dfs = {result.path.split("/")[-1]: result.metrics_dataframe for result in results}
    best_result = results.get_best_result("val_loss_gen", "min")
    print("Best config is:", best_result.metrics, best_result.config)

    if plot:
        ax = None  # This plots everything on the same plot
        for d in dfs.values():
            ax = d.val_loss_gen.plot(ax=ax, legend=False, logy=True)
        # ax.set_ylim([-10, 10])
        import matplotlib.pyplot as plt
        plt.show()

    return results.get_best_result("val_loss_gen", "min").config, dfs


def train_opf(model, train_loader, val_loader, max_epochs=200, y_nodes=["gen", "ext_grid"], log_every=10,
              device="cpu", decay_lr=0.3, hetero=True, base_lr=0.01, experiment=None, clamp_boundary=0,
              use_physical_loss=1, weighting="relative"):
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    milestones = [max_epochs // 2, (max_epochs * 3) // 4, (max_epochs * 9) // 10]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=decay_lr)

    loss_fn = torch.nn.MSELoss(reduction="none")
    loss_fn = node_loss

    train_losses = []
    boundary_train_losses = []
    physical_train_losses = []
    boundary_val_losses = []
    physical_val_losses = []
    val_losses = []
    val_losses_bus = []
    val_losses_line = []
    val_losses_gen = []
    val_losses_ext_grid = []
    learning_rate = []
    neighboorhood = None

    for epoch in range(0, max_epochs):
        lr = lr_scheduler.get_last_lr()[0]
        learning_rate.append(lr)

        train_loss = 0
        boundary_train_loss = 0
        physical_train_loss = 0

        for batch_id, batch in enumerate(train_loader):
            out, labels, loss, losses, b_losses, p_losses, neighboorhood = train_step(model, optimizer, batch, None,
                    y_nodes, loss_fn, hetero, clamp_boundary=(clamp_boundary == 1 or clamp_boundary == 2),
                  use_physical_loss=use_physical_loss, neighboorhood=neighboorhood, epoch=epoch,batch_id=batch_id,
                                                                                      weighting=weighting)
            train_loss += loss
            boundary_train_loss += np.concatenate(b_losses, 0).max() if len(b_losses) else 0
            physical_train_loss += np.concatenate(p_losses, 0).mean() if len(p_losses) else 0

        lr_scheduler.step()
        train_loss /= len(train_loader)
        boundary_train_loss /= len(train_loader)
        physical_train_loss /= len(train_loader)

        if epoch % log_every == 0:
            print("epoch", epoch)
            print("training loss", train_loss)
            print("boundary loss", boundary_train_loss)
            print("physical loss", physical_train_loss)

        train_losses.append(train_loss)
        boundary_train_losses.append(boundary_train_loss)
        physical_train_losses.append(physical_train_loss)

        val_loss = 0
        boundary_loss_val = 0
        physical_loss_val = 0
        val_loss_gen = 0
        val_loss_ext_grid = 0
        val_loss_bus = 0
        val_loss_line = 0

        val_losses_all = []
        out_all = []
        labels_all = []
        with torch.no_grad():
            for batch_id, batch in enumerate(val_loader):
                last_out, last_label, loss, losses, b_losses, p_losses, neighboorhood = eval_step(model, batch, None,
                  y_nodes, loss_fn, hetero, clamp_boundary=(clamp_boundary == 2 or clamp_boundary == 3),
                   use_physical_loss=use_physical_loss,neighboorhood=neighboorhood, epoch=epoch, batch_id=batch_id)
                val_loss += loss
                boundary_loss_val += np.concatenate(b_losses, 0).max() if len(b_losses) else 0
                physical_loss_val += np.concatenate(b_losses, 0).mean() if len(p_losses) else 0
                val_losses_all.append(losses)
                out_all.append(last_out)
                labels_all.append(last_label)
                val_loss_gen += losses[0].mean()
                val_loss_ext_grid += losses[1].mean() if len(losses) > 1 else 0
                val_loss_bus += losses[2].mean() if len(losses) > 2 else 0
                val_loss_line += losses[3].mean() if len(losses) > 3 else 0

        val_loss /= len(val_loader)
        boundary_loss_val /= len(val_loader)
        physical_loss_val /= len(val_loader)
        if epoch % log_every == 0:
            print("validation loss", val_loss)
            print("boundary loss", boundary_loss_val)

        val_losses.append(val_loss)
        boundary_val_losses.append(boundary_loss_val)
        physical_train_losses.append(physical_loss_val)

        val_loss_gen /= len(val_loader)
        val_losses_gen.append(val_loss_gen)

        val_loss_ext_grid /= len(val_loader)
        val_losses_ext_grid.append(val_loss_ext_grid)

        val_loss_bus /= len(val_loader)
        val_losses_bus.append(val_loss_bus)

        val_loss_line /= len(val_loader)
        val_losses_line.append(val_loss_line)

        if experiment is not None:
            log_dict = {"train_losses": train_loss,
                        "val_losses": val_loss,"p_train_losses":physical_train_loss,"p_val_losses":physical_loss_val,
                        "b_train_losses": boundary_train_loss, "b_val_losses": boundary_loss_val, "learning_rate": lr,
                        "val_losses_gen": val_loss_gen, "val_losses_ext_grid": val_loss_ext_grid,
                        "val_losses_bus": val_loss_bus, "val_losses_line": val_loss_line}
            experiment.log_metrics(log_dict, epoch=epoch)
            logged_metrics = {}
            outputs_keys=  {"gen":[0,1],"ext_grid":[2,3],"bus":[3,4]}
            for k, v in out_all[0].items():
                logged_metrics = {**logged_metrics,
                                  **{f"out_{k}_{i}_0": val.cpu().item() for (i, val) in enumerate(v[0]) if i in outputs_keys.get(k)}}
                logged_metrics = {**logged_metrics,
                                  **{f"out_{k}_{i}_1": val.cpu().item() for (i, val) in enumerate(v[1]) if
                                     i in outputs_keys.get(k)}}

                logged_metrics = {**logged_metrics,
                                  **{f"label_{k}_{i}_0": val.cpu().item() for (i, val) in enumerate(labels_all[0][k][0])}}
                logged_metrics = {**logged_metrics,
                                  **{f"label_{k}_{i}_1": val.cpu().item() for (i, val) in enumerate(labels_all[0][k][1])}}


            experiment.log_metrics(logged_metrics, epoch=epoch)

    print("Training over")
    return train_losses, val_losses, (val_losses_gen, val_losses_ext_grid, val_losses_bus, val_losses_line), (
        out_all, val_losses_all), boundary_train_losses, physical_train_losses, boundary_val_losses, learning_rate


def train_step(model, optimizer, data, mask_node="paper", feature_node="paper", loss_f=None, hetero=True,
               use_boundary_loss=True, clamp_boundary=True, use_physical_loss=1, neighboorhood=None
               , epoch=0, batch_id=0, weighting="relative"):
    model.train()
    optimizer.zero_grad()
    if loss_f is None:
        loss_f = F.mse_loss

    if isinstance(feature_node, str):
        feature_node = [feature_node]
    if isinstance(mask_node, str):
        mask_node = [mask_node]

    loss = 0
    losses = []
    boundary_losses = []
    physical_losses = []
    return_label = {}
    return_output = {}
    if hetero:
        out = model(data.x_dict, data.edge_index_dict)
        total_nodes = {node:len(data[node].y) for node in feature_node}
        for i, node in enumerate(feature_node):
            label = data[node].y
            output = out[node]
            if mask_node is not None:
                mask = data[mask_node[i]].train_mask
                label = label[mask_node[i]]
                output = output[mask_node[i]]

            output_mask = data[node].output_mask

            loss_label = loss_f(label, output, output_mask)
            losses.append(loss_label.cpu().detach().numpy())
            if clamp_boundary:
                output = clamp_boundaries(data[node].boundaries, output, node, output_mask)

            return_output[node] = output
            return_label[node] = label

            if use_boundary_loss:
                loss_boundary = boundary_loss(data[node].boundaries, output, node=node)
                loss_node = torch.cat([loss_label.reshape((loss_boundary.shape[0], -1)), loss_boundary.unsqueeze(1)], 1)
                boundary_losses.append(loss_boundary.cpu().detach().numpy())
            else:
                loss_node = loss_label

            weight_node = np.sum(list(total_nodes.values()))/total_nodes.get(node) if weighting=="relative" else 1
            loss += weight_node * loss_node.mean()

            if use_physical_loss and node=="bus":
                physical_loss, neighboorhood = power_imbalance_loss(data, out, neighboorhood)
                physical_losses.append(physical_loss.sum(1).detach().numpy())
                if use_physical_loss == 2:
                    loss += physical_loss.sum()
                elif use_physical_loss == 3:
                    loss = physical_loss.sum()

    else:
        mask = ~torch.isnan(data.y).any(1)
        label = data.y[mask]
        if isinstance(model, GNN):
            out = model(data.x, data.edge_index)
            output = out[mask]
            return_output = output
        else:
            x = data.x.reshape(data.batch_size, -1)
            label = label.reshape(data.batch_size, -1)
            return_output = output = model(x)
        loss_label = loss_f(label, output)
        # loss_boundary = boundary_loss(data[node].boundaries, output)
        loss_node = loss_label  # torch.cat([loss_label,loss_boundary.unsqueeze(1)],1)
        losses.append(loss_label.cpu().detach().numpy())
        # boundary_losses.append(loss_boundary.cpu().detach().numpy())
        loss += loss_node.mean()

    loss.backward()
    optimizer.step()

    return return_output, return_label, float(loss), losses, boundary_losses, physical_losses, neighboorhood


def timeit(model, data, count=100000, hetero=True):
    import time
    return
    nb = count // len(data)
    params = (data.x_dict, data.edge_index_dict) if hetero else (data.x, data.edge_index)
    begin = time.time()

    for a in range(nb):
        model(*params)
    total = time.time() - begin
    print(total)


def eval_step(model, data, mask_node="paper", feature_node="paper", loss_f=None, hetero=True, clamp_boundary=0,
              use_physical_loss=1, neighboorhood=None, epoch=0, batch_id=0):
    model.eval()
    if loss_f is None:
        loss_f = F.cross_entropy

    if isinstance(feature_node, str):
        feature_node = [feature_node]
    if isinstance(mask_node, str):
        mask_node = [mask_node]
    loss = 0
    losses = []
    boundary_losses = []
    physical_losses = []

    return_label = {}
    return_output = {}
    if hetero:
        out = model(data.x_dict, data.edge_index_dict)

        timeit(model, data)

        for i, node in enumerate(feature_node):
            label = data[node].y
            output = out[node]
            if mask_node is not None:
                mask = data[mask_node[i]].train_mask
                label = label[mask_node[i]]
                output = output[mask_node[i]]

            output_mask = data[node].output_mask
            loss_node = loss_f(label, output, output_mask)
            loss_boundary = boundary_loss(data[node].boundaries, output,node=node)

            if clamp_boundary:
                return_output[node] = clamp_boundaries(data[node].boundaries, output, node, output_mask)
            else:
                return_output[node] = output

            return_label[node] = label
            losses.append(loss_node.cpu().detach().numpy())
            boundary_losses.append(loss_boundary.cpu().detach().numpy())
            loss += loss_node.sum()

            if use_physical_loss and node=="bus":
                physical_loss, neighboorhood = power_imbalance_loss(data, out,neighboorhood=neighboorhood)
                physical_losses.append(physical_loss.sum().cpu().detach().numpy())

    else:
        mask = ~torch.isnan(data.y).any(1)
        label = data.y[mask]
        if isinstance(model, GNN):
            out = model(data.x, data.edge_index)
            timeit(model, data, hetero=False)
            output = out[mask]
            out = output
        else:
            x = data.x.reshape(data.batch_size, -1)
            original_shape = label.shape
            label = label.reshape(data.batch_size, -1)
            output = model(x)
            out = output.reshape(original_shape)
        return_output = out
        loss_node = loss_f(label, output)
        # loss_boundary = boundary_loss(data[node].boundaries, output)
        losses.append(loss_node.cpu().detach().numpy())
        # boundary_losses.append(loss_boundary.cpu().detach().numpy())
        loss += loss_node.mean()

    return return_output, return_label, float(loss), losses, boundary_losses, physical_losses, neighboorhood
