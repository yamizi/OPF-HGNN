import torch.nn.functional as F

def train_step(model, optimizer, data, mask_node="paper", feature_node="paper", loss_f=None):
    model.train()
    optimizer.zero_grad()
    if loss_f is None:
        loss_f = F.cross_entropy
    
    if isinstance(feature_node,str):
        feature_node = [feature_node]
    if isinstance(mask_node,str):
        mask_node = [mask_node]

    out = model(data.x_dict, data.edge_index_dict)
    loss = 0
    losses = []
    for i, node in enumerate(feature_node):
        label = data[node].y
        output = out[node]
        if mask_node is not None:
            mask = data[mask_node[i]].train_mask
            label = label[mask_node[i]]
            output = output[mask_node[i]]

        loss_node = loss_f(label, output)
        losses.append(loss_node.item())
        loss += loss_node

    loss.backward()
    optimizer.step()
    return out, float(loss), losses


def eval_step(model, data, mask_node="paper", feature_node="paper", loss_f=None):
    model.eval()
    if loss_f is None:
        loss_f = F.cross_entropy

    if isinstance(feature_node,str):
        feature_node = [feature_node]
    if isinstance(mask_node,str):
        mask_node = [mask_node]

    out = model(data.x_dict, data.edge_index_dict)
    loss = 0
    losses = []
    for i, node in enumerate(feature_node):    
        label = data[node].y
        output = out[node]
        if mask_node is not None:
            mask = data[mask_node[i]].train_mask
            label = label[mask_node[i]]
            output = output[mask_node[i]]

        loss_node = loss_f(label, output)
        losses.append(loss_node.item())
        loss += loss_node

    return out, float(loss),  losses