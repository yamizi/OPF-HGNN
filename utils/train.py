import torch.nn.functional as F
import torch

def relative_loss(yhat,y):
    criterion = torch.nn.L1Loss(reduction="none")
    #criterion = torch.nn.MSELoss(reduction="none")
    return criterion(yhat,y)/yhat.abs()

def boundary_loss(boundaries,y):
    minp = boundaries[:,0]
    maxp = boundaries[:,1]
    minq = boundaries[:,2]
    maxq = boundaries[:,3]
    return torch.max(torch.zeros_like(minp),minp-y[:,0]) + torch.max(torch.zeros_like(maxp),y[:,0]-maxp) + torch.max(torch.zeros_like(minq),minq-y[:,1]) + torch.max(torch.zeros_like(maxq),y[:,1]-maxq)

def train_opf(model,train_loader, val_loader, max_epochs=200, y_nodes=["gen","ext_grid"], log_every=10,
              device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss() 
    loss_fn = relative_loss

    train_losses = []
    val_losses = []
    val_losses_gen = []
    val_losses_ext_grid  = []
    

    for epoch in range(0,max_epochs):
        train_loss = 0
        
        for batch in train_loader:
            batch = batch
            out, loss, losses = train_step(model, optimizer,batch,None,y_nodes,loss_fn)
            train_loss += loss

        train_loss /= len(train_loader)
        if epoch % log_every == 0:
            print("epoch",epoch)
            print("training loss",train_loss)
        train_losses.append(train_loss)

        val_loss = 0
        val_loss_gen = 0
        val_loss_ext_grid = 0

        val_losses_all = []
        out_all = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                last_out, loss, losses = eval_step(model, batch,None,y_nodes,loss_fn)
                val_loss += loss
                val_losses_all.append(losses)
                out_all.append(last_out)
                val_loss_gen += losses[0].mean()
                val_loss_ext_grid += losses[1].mean()

        val_loss /= len(val_loader)
        if epoch % log_every == 0:
            print("validation loss",val_loss)
            
        val_losses.append(val_loss)
        
        val_loss_gen /= len(val_loader)
        val_losses_gen.append(val_loss_gen)

        val_loss_ext_grid /= len(val_loader)
        val_losses_ext_grid.append(val_loss_ext_grid)

    return train_losses, val_losses, val_losses_gen, val_losses_ext_grid, (out_all, val_losses_all)

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

        loss_label = loss_f(label, output)
        loss_boundary = boundary_loss(data[node].boundaries, output)
        loss_node = torch.cat([loss_label,loss_boundary.unsqueeze(1)],1)
        losses.append(loss_node.cpu().detach().numpy())
        loss += loss_node.mean()

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
        losses.append(loss_node.cpu().detach().numpy())
        loss += loss_node.mean()

    return out, float(loss),  losses