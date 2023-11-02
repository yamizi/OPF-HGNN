import torch.nn.functional as F
import torch

def relative_loss(yhat,y):
    #criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    return criterion(yhat,y)/yhat

def train_opf(model,train_loader, val_loader, max_epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    
    train_losses = []
    val_losses = []
    val_losses_gen = []
    val_losses_ext_grid  = []

    for epoch in range(1,max_epochs):
        train_loss = 0
        print("epoch",epoch)
        for batch in train_loader:
            out, loss, losses = train_step(model, optimizer,batch,None,["gen","ext_grid"],relative_loss)
            train_loss += loss

        train_loss /= len(train_loader)
        print("training loss",train_loss)
        train_losses.append(train_loss)

        val_loss = 0
        val_loss_gen = 0
        val_loss_ext_grid = 0
        
        for batch in val_loader:
            last_out, loss, losses = eval_step(model, batch,None,["gen","ext_grid"],relative_loss)
            val_loss += loss
            val_loss_gen += losses[0]
            val_loss_ext_grid += losses[1]

        val_loss /= len(val_loader)
        print("validation loss",val_loss)
        val_losses.append(val_loss)
        
        val_loss_gen /= len(val_loader)
        val_losses_gen.append(val_loss_gen)

        val_loss_ext_grid /= len(val_loader)
        val_losses_ext_grid.append(val_loss_ext_grid)

    return train_losses, val_losses, val_losses_gen, val_losses_ext_grid, last_out

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