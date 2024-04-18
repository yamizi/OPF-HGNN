import torch
import numpy as np
from utils.pandapower.normalization import MAX_ANGLE
import time
from torch.nn.functional import mse_loss


def relative_loss(yhat, y):
    criterion = torch.nn.L1Loss(reduction="none")
    # criterion = torch.nn.MSELoss(reduction="none")
    return criterion(yhat, y) / yhat.abs()


def node_loss(yhat, y, mask=None):
    criterions = [torch.nn.MSELoss(reduction="none"), torch.nn.L1Loss(reduction="none")]
    if mask is None:
        losses = [criterion(yhat, y[:, :yhat.shape[1]]).unsqueeze(0) for criterion in criterions]
    else:
        losses = [criterion(yhat.flatten(), y.flatten()[mask.bool().flatten()]).unsqueeze(0) for criterion in
                  criterions]

    return torch.cat(losses).sum(0)


def boundary_loss(boundaries, y, node=""):
    # minp = boundaries[:,0] # maxp = boundaries[:,1]
    # minq = boundaries[:,2] # maxq = boundaries[:,3]
    # minVm = boundaries[:,4] # maxVm = boundaries[:,5]
    # minVa = boundaries[:,6] # maxVa = boundaries[:,7]
    # mini_from_ka = boundaries[:,12] # maxi_from_ka = boundaries[:,13]
    # mini_to_ka = boundaries[:,14] # maxi_to_ka = boundaries[:,15]

    boundary_losses = [torch.max(torch.zeros_like(boundaries[:, 2 * i]), boundaries[:, 2 * i] - y[:, i]) + torch.max(
        torch.zeros_like(boundaries[:, 2 * i + 1]), y[:, i] - boundaries[:, 2 * i + 1]) for i in range(y.shape[1])
                       if torch.isnan(boundaries[:, 2 * i]).sum() == 0]
    return torch.stack(boundary_losses).sum(0)
    # return torch.max(torch.zeros_like(minp),minp-y[:,0]) + torch.max(torch.zeros_like(maxp),y[:,0]-maxp) + torch.max(torch.zeros_like(minq),minq-y[:,1]) + torch.max(torch.zeros_like(maxq),y[:,1]-maxq)


def power_imbalance_loss(data, out, neighboorhood=None, ground_truth=False):
    """calculate injected power Pji

    Formula:
    $$
    P_{ji} = V_m^i*V_m^j*Y_{ij}*\cos(V_a^i-V_a^j-\theta_{ij})
            -(V_m^i)^2*Y_{ij}*\cos(-\theta_{ij})
    $$
    $$
    Q_{ji} = V_m^i*V_m^j*Y_{ij}*\sin(V_a^i-V_a^j-\theta_{ij})
            -(V_m^i)^2*Y_{ij}*\sin(-\theta_{ij})
    $$

    Input:
    data: the GNN features and labels
    output: the ouput of the GCN, P and Q for genertors and external grid and Vm,Va for loads
    ground_truth: True to compute the losses on the ground truth, should return close to 0 (for debugging purpose)

    Return:
        MSE loss on Pi and Qi between the two terms of the loss

    """
    begin = time.time()
    durations = []

    ## Caching repetitive variables that remain the same in all batches (except last one when incomplete)
    if neighboorhood is None or len(neighboorhood[2]) != len(out.get("gen")):

        bus_to_line_list = list(zip(*data.edge_index_dict.get(('bus', 'to', 'line')).cpu().tolist()))
        line_to_bus_list = list(zip(*data.edge_index_dict.get(('line', 'to', 'bus')).cpu().tolist()))

        bus_to_gen_index, gen_index = data.edge_index_dict.get(('bus', 'to', 'gen')).cpu().tolist()
        bus_to_ext_index, ext_index = data.edge_index_dict.get(('bus', 'to', 'ext_grid')).cpu().tolist()

        bus_to_bus_dict = [
            (start, end, l1) for
            (start, l1) in bus_to_line_list for (l2, end) in line_to_bus_list if (l1 == l2 and start != end)]

        i, j = [a[0] for a in bus_to_bus_dict], [a[1] for a in bus_to_bus_dict]

        bus_size = len(out.get("bus"))
        buses = torch.LongTensor(i)

        mask = torch.stack([buses == a for a in np.arange(bus_size)])

        neighboorhood = bus_to_line_list, line_to_bus_list, bus_to_gen_index, gen_index, bus_to_ext_index, ext_index, bus_to_bus_dict, i, j, mask

    else:
        bus_to_line_list, line_to_bus_list, bus_to_gen_index, gen_index, bus_to_ext_index, ext_index, bus_to_bus_dict, i, j, mask = neighboorhood

    # line features are: 'std_type', 'length_km', 'r_ohm_per_km', 'x_ohm_per_km', 'c_nf_per_km','g_us_per_km'
    line_features = data.x_dict.get("line")[:, :6]
    # bus features are : 'vn_kv', 'in_service', 'min_vm_pu', 'max_vm_pu', 'p_mw', 'q_mvar'
    bus_features = data.x_dict.get("bus")[:, -2:]

    # (i, j, r_ij, x_ij)
    r, x = torch.stack([line_features[a[2], 1] * line_features[a[2], 2] for a in bus_to_bus_dict]), torch.stack(
        [line_features[a[2], 1] * line_features[a[2], 3] for a in bus_to_bus_dict])

    if ground_truth:
        vm_i = data["bus"].y[i, 0]
        vm_j = data["bus"].y[j, 0]

        va_i = MAX_ANGLE / 180. * torch.pi * data["bus"].y[i, 1]
        va_j = MAX_ANGLE / 180. * torch.pi * data["bus"].y[j, 1]

        gen_buses = data["gen"].y[gen_index, :]
        ext_buses = data["ext_grid"].y[ext_index, :]
    else:
        vm_i = out.get("bus")[i, 4]
        vm_j = out.get("bus")[j, 4]

        va_i = MAX_ANGLE / 180. * torch.pi * out.get("bus")[i, 5]
        va_j = MAX_ANGLE / 180. * torch.pi * out.get("bus")[j, 5]

        gen_buses = out.get("gen")[gen_index, 0:2]
        ext_buses = out.get("ext_grid")[ext_index, 2:4]

    ## First side of the loss
    g_ij = r / (r ** 2 + x ** 2)
    b_ij = -x / (r ** 2 + x ** 2)

    ###### Va in label is pre-normalized by division over 50 (cf # Normalize angles in pandapower_graph.py)
    e_i = vm_i * torch.cos(va_i)
    f_i = vm_i * torch.sin(va_i)
    e_j = vm_j * torch.cos(va_j)
    f_j = vm_j * torch.sin(va_j)

    #### PowerflowNet ######
    Pji = g_ij * (e_i * e_j - e_i ** 2 + f_i * f_j - f_i ** 2) + b_ij * (f_i * e_j - e_i * f_j)
    Qji = g_ij * (f_i * e_j - e_i * f_j) + b_ij * (-e_i * e_j + e_i ** 2 - f_i * f_j + f_i ** 2)

    ##### merging them by node i for each bus

    Pi = torch.matmul(mask.float(), Pji)
    Qi = torch.matmul(mask.float(), Qji)

    ## Second side of the loss

    bus_generator = torch.zeros_like(bus_features)
    bus_ext = torch.zeros_like(bus_features)

    # Step by step
    # predicted_gen_P = data.sn_mva[0] * out.get("gen")[gen_index, 0]
    # predicted_gen_Q = data.sn_mva[0] * out.get("gen")[gen_index, 1]
    # predicted_ext_P = data.sn_mva[0] * out.get("ext_grid")[ext_index, 2]
    # predicted_ext_Q = data.sn_mva[0] * out.get("ext_grid")[ext_index, 3]
    # bus_generator[bus_to_gen_index, 0] = predicted_gen_P
    # bus_generator[bus_to_gen_index, 1] = predicted_gen_Q
    # bus_ext[bus_to_ext_index, 0] = predicted_ext_P
    # bus_ext[bus_to_ext_index, 1] = predicted_ext_Q

    # Equivalent
    bus_generator[bus_to_gen_index, :] = data.sn_mva[0] * gen_buses
    bus_ext[bus_to_ext_index, :] = data.sn_mva[0] * ext_buses

    bus_true = bus_features + bus_generator + bus_ext
    Pi_true = bus_true[:, 0]
    Qi_true = bus_true[:, 1]

    duration = time.time() - begin
    loss = [mse_loss(Pi, Pi_true).unsqueeze(0), mse_loss(Qi, Qi_true).unsqueeze(0)]

    return torch.cat(loss, dim=-1), neighboorhood, duration  # (num_edges, 2)
