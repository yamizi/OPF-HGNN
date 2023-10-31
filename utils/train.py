import torch.nn.functional as F

def train_step(model, optimizer, data, mask_node="paper", feature_node="paper", loss_f=None):
    model.train()
    optimizer.zero_grad()
    if loss_f is None:
        loss_f = F.cross_entropy
    out = model(data.x_dict, data.edge_index_dict)
    
    label = data[feature_node].y
    output = out[feature_node]
    if mask_node is not None:
        mask = data[mask_node].train_mask
        label = label[mask]
        output = output[mask]

    loss = loss_f(label, output)
    loss.backward()
    optimizer.step()
    return out, float(loss)


def eval_step(model, data, mask_node="paper", feature_node="paper", loss_f=None):
    model.eval()
    if loss_f is None:
        loss_f = F.cross_entropy
    out = model(data.x_dict, data.edge_index_dict)
    
    label = data[feature_node].y
    output = out[feature_node]
    if mask_node is not None:
        mask = data[mask_node].train_mask
        label = label[mask]
        output = output[mask]

    loss = loss_f(label, output)
    return out, float(loss)