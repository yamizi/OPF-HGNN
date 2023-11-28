import torch.nn.functional as F
import torch
import numpy as np
from utils.models import FCNN, GNN

def relative_loss(yhat,y):
    criterion = torch.nn.L1Loss(reduction="none")
    #criterion = torch.nn.MSELoss(reduction="none")
    return criterion(yhat,y)/yhat.abs()

def masked_loss(yhat,y):
    criterion = torch.nn.MSELoss(reduction="none")
    return criterion(yhat,y[:,:yhat.shape[1]])

def boundary_loss(boundaries,y, node=""):
    #minp = boundaries[:,0]
    #maxp = boundaries[:,1]
    #minq = boundaries[:,2]
    #maxq = boundaries[:,3]

    boundary_losses = [torch.max(torch.zeros_like(boundaries[:,2*i]),boundaries[:,2*i]-y[:,i]) + torch.max(torch.zeros_like(boundaries[:,2*i+1]),y[:,i]-boundaries[:,2*i+1]) for i in range(y.shape[1]-1) if torch.isnan(boundaries[:,2*i]).sum()==0]
    return torch.stack(boundary_losses).sum(0)
    #return torch.max(torch.zeros_like(minp),minp-y[:,0]) + torch.max(torch.zeros_like(maxp),y[:,0]-maxp) + torch.max(torch.zeros_like(minq),minq-y[:,1]) + torch.max(torch.zeros_like(maxq),y[:,1]-maxq)

def train_opf(model,train_loader, val_loader, max_epochs=200, y_nodes=["gen","ext_grid"], log_every=10,
              device="cpu",decayRate = 0.3, hetero=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    milestones=[max_epochs//2,(max_epochs*3)//4,(max_epochs*9)//10]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones,gamma=decayRate)

    loss_fn = torch.nn.MSELoss(reduction="none")
    loss_fn = masked_loss

    train_losses = []
    boundary_train_losses = []
    boundary_val_losses = []
    val_losses = []
    val_losses_bus = []
    val_losses_gen = []
    val_losses_ext_grid  = []
    learning_rate = []

    for epoch in range(0,max_epochs):
        lr = lr_scheduler.get_last_lr()[0]
        learning_rate.append(lr)

        train_loss = 0
        boundary_loss = 0
        for batch in train_loader:
            out, loss, losses, b_losses = train_step(model, optimizer,batch,None,y_nodes,loss_fn, hetero)
            train_loss += loss
            boundary_loss+= np.concatenate(b_losses,0).max() if len(b_losses) else 0

        lr_scheduler.step()
        train_loss /= len(train_loader)
        boundary_loss /= len(train_loader)

        if epoch % log_every == 0:
            print("epoch",epoch)
            print("training loss",train_loss)
            print("boundary loss",boundary_loss)

        train_losses.append(train_loss)
        boundary_train_losses.append(boundary_loss)

        val_loss = 0
        boundary_loss = 0
        val_loss_gen = 0
        val_loss_ext_grid = 0
        val_loss_bus = 0

        val_losses_all = []
        out_all = []
        with torch.no_grad():
            for batch in val_loader:
                last_out, loss, losses, b_losses = eval_step(model, batch,None,y_nodes,loss_fn, hetero)
                val_loss += loss
                boundary_loss+= np.concatenate(b_losses,0).max() if len(b_losses) else 0
                val_losses_all.append(losses)
                out_all.append(last_out)
                val_loss_gen += losses[0].mean()
                val_loss_ext_grid += losses[1].mean() if len(losses)>1 else 0
                val_loss_bus += losses[2].mean() if len(losses) > 2 else 0

        val_loss /= len(val_loader)
        boundary_loss /= len(val_loader)
        if epoch % log_every == 0:
            print("validation loss",val_loss)
            print("boundary loss",boundary_loss)
            
        val_losses.append(val_loss)
        boundary_val_losses.append(boundary_loss)
        
        val_loss_gen /= len(val_loader)
        val_losses_gen.append(val_loss_gen)

        val_loss_ext_grid /= len(val_loader)
        val_losses_ext_grid.append(val_loss_ext_grid)

        val_loss_bus /= len(val_loader)
        val_losses_bus.append(val_loss_bus)

    print("Training over")
    return train_losses, val_losses, (val_losses_gen, val_losses_ext_grid,val_losses_bus), (out_all, val_losses_all), boundary_train_losses, boundary_val_losses, learning_rate

def train_step(model, optimizer, data, mask_node="paper", feature_node="paper", loss_f=None, hetero=True, 
               use_boundary_loss=True):
    model.train()
    optimizer.zero_grad()
    if loss_f is None:
        loss_f = F.cross_entropy
    
    if isinstance(feature_node,str):
        feature_node = [feature_node]
    if isinstance(mask_node,str):
        mask_node = [mask_node]

    loss = 0
    losses = []
    boundary_losses = []

    if hetero:
        out = model(data.x_dict, data.edge_index_dict)
        
        for i, node in enumerate(feature_node):
            label = data[node].y
            output = out[node]
            if mask_node is not None:
                mask = data[mask_node[i]].train_mask
                label = label[mask_node[i]]
                output = output[mask_node[i]]

            loss_label = loss_f(label, output)
            losses.append(loss_label.cpu().detach().numpy())
            if use_boundary_loss:
                loss_boundary = boundary_loss(data[node].boundaries, output, node)
                loss_node = torch.cat([loss_label,loss_boundary.unsqueeze(1)],1)
                boundary_losses.append(loss_boundary.cpu().detach().numpy())
            else:
                loss_node = loss_label
            loss += loss_node.mean()
    else:
        mask = ~torch.isnan(data.y).any(1)
        label = data.y[mask]
        if isinstance(model, GNN): 
            out =  model(data.x, data.edge_index)
            output = out[mask]
            out = output
        else:
            x = data.x.reshape(data.batch_size,-1)
            label = label.reshape(data.batch_size,-1)
            out = output = model(x)
        loss_label = loss_f(label, output)
        #loss_boundary = boundary_loss(data[node].boundaries, output)
        loss_node = loss_label # torch.cat([loss_label,loss_boundary.unsqueeze(1)],1)
        losses.append(loss_label.cpu().detach().numpy())
        #boundary_losses.append(loss_boundary.cpu().detach().numpy())
        loss += loss_node.mean()

    loss.backward()
    optimizer.step()
    
    return out, float(loss), losses, boundary_losses


def eval_step(model, data, mask_node="paper", feature_node="paper", loss_f=None, hetero=True):
    model.eval()
    if loss_f is None:
        loss_f = F.cross_entropy

    if isinstance(feature_node,str):
        feature_node = [feature_node]
    if isinstance(mask_node,str):
        mask_node = [mask_node]
    loss = 0
    losses = []
    boundary_losses = []

    if hetero:
        out = model(data.x_dict, data.edge_index_dict)
        
        for i, node in enumerate(feature_node):    
            label = data[node].y
            output = out[node]
            if mask_node is not None:
                mask = data[mask_node[i]].train_mask
                label = label[mask_node[i]]
                output = output[mask_node[i]]

            loss_node = loss_f(label, output)
            loss_boundary = boundary_loss(data[node].boundaries, output)
            losses.append(loss_node.cpu().detach().numpy())
            boundary_losses.append(loss_boundary.cpu().detach().numpy())
            loss += loss_node.mean()
    else:
        mask = ~torch.isnan(data.y).any(1)
        label = data.y[mask]
        if isinstance(model, GNN):
            out = model(data.x, data.edge_index)
            output = out[mask]
            out = output
        else:
            x = data.x.reshape(data.batch_size, -1)
            original_shape = label.shape
            label = label.reshape(data.batch_size, -1)
            output = model(x)
            out = output.reshape(original_shape)

        loss_node = loss_f(label, output)
        #loss_boundary = boundary_loss(data[node].boundaries, output)
        losses.append(loss_node.cpu().detach().numpy())
        #boundary_losses.append(loss_boundary.cpu().detach().numpy())
        loss += loss_node.mean()

    return out, float(loss),  losses, boundary_losses