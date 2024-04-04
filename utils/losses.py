import torch
def relative_loss(yhat, y):
    criterion = torch.nn.L1Loss(reduction="none")
    # criterion = torch.nn.MSELoss(reduction="none")
    return criterion(yhat, y) / yhat.abs()


def node_loss(yhat, y, mask=None):
    criterion = torch.nn.MSELoss(reduction="none")
    if mask is None:
        return criterion(yhat, y[:, :yhat.shape[1]])
    else:
        return criterion(yhat.flatten(), y.flatten()[mask.bool().flatten()])




def boundary_loss(boundaries, y, node=""):
    # minp = boundaries[:,0] # maxp = boundaries[:,1]
    # minq = boundaries[:,2] # maxq = boundaries[:,3]
    # minVm = boundaries[:,4] # maxVm = boundaries[:,5]
    # minVa = boundaries[:,6] # maxVa = boundaries[:,7]
    # mini_from_ka = boundaries[:,12] # maxi_from_ka = boundaries[:,13]
    # mini_to_ka = boundaries[:,14] # maxi_to_ka = boundaries[:,15]

    boundary_losses = [torch.max(torch.zeros_like(boundaries[:, 2 * i]), boundaries[:, 2 * i] - y[:, i]) + torch.max(
        torch.zeros_like(boundaries[:, 2 * i + 1]), y[:, i] - boundaries[:, 2 * i + 1]) for i in range(y.shape[1] - 1)
                       if torch.isnan(boundaries[:, 2 * i]).sum() == 0]
    return torch.stack(boundary_losses).sum(0)
    # return torch.max(torch.zeros_like(minp),minp-y[:,0]) + torch.max(torch.zeros_like(maxp),y[:,0]-maxp) + torch.max(torch.zeros_like(minq),minq-y[:,1]) + torch.max(torch.zeros_like(maxq),y[:,1]-maxq)


def  power_imbalance_loss(self, x_i, x_j, edge_attr):
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
            x_i: (num_edges, 6)
            x_j: (num_edges, 6)
            edge_attr: (num_edges, 2)

        Return:
            Pji|Qji: (num_edges, 2)
        """
        r_x = edge_attr[:, 0:2] # (num_edges, 2)
        r, x = r_x[:, 0:1], r_x[:, 1:2]
        # zm_ij = torch.norm(r_x, p=2, dim=-1, keepdim=True) # (num_edges, 1) NOTE (r**2+x**2)**0.5 should be non-zero
        # za_ij = torch.acos(edge_attr[:, 0:1] / zm_ij) # (num_edges, 1)
        # ym_ij = 1/(zm_ij + 1e-6)        # (num_edges, 1)
        # ya_ij = -za_ij      # (num_edges, 1)
        # g_ij = ym_ij * torch.cos(ya_ij) # (num_edges, 1)
        # b_ij = ym_ij * torch.sin(ya_ij) # (num_edges, 1)
        g_ij = r / (r**2 + x**2)
        b_ij = -x / (r**2 + x**2)
        ym_ij = torch.sqrt(g_ij**2+b_ij**2)
        ya_ij = torch.acos(g_ij/ym_ij)
        vm_i = x_i[:, 0:1] # (num_edges, 1)
        va_i = 1/180.*torch.pi*x_i[:, 1:2] # (num_edges, 1)
        vm_j = x_j[:, 0:1] # (num_edges, 1)
        va_j = 1/180.*torch.pi*x_j[:, 1:2] # (num_edges, 1)
        e_i = vm_i * torch.cos(va_i)
        f_i = vm_i * torch.sin(va_i)
        e_j = vm_j * torch.cos(va_j)
        f_j = vm_j * torch.sin(va_j)

        ####### my (incomplete) method #######
        # Pji = vm_i * vm_j * ym_ij * torch.cos(va_i - va_j - ya_ij) \
        #         - vm_i**2 * ym_ij * torch.cos(-ya_ij)
        # Qji = vm_i * vm_j * ym_ij * torch.sin(va_i - va_j - ya_ij) \
        #         - vm_i**2 * ym_ij * torch.sin(-ya_ij)

        ####### standard method #######
        # cannot be done since there's not complete information about whole neighborhood.

        ####### another reference method #######
        # Pji = vm_i * vm_j * (g_ij*torch.cos(va_i-va_j)+b_ij*torch.sin(va_i-va_j))
        # Qji = vm_i * vm_j * (g_ij*torch.sin(va_i-va_j)-b_ij*torch.cos(va_i-va_j))

        ####### reference method 3 #######
        # Pji = g_ij*(vm_i**2 - vm_i*vm_j*torch.cos(va_i-va_j)) \
        #     - b_ij*(vm_i*vm_j*torch.sin(va_i-va_j))
        # Qji = b_ij*(- vm_i**2 + vm_i*vm_j*torch.cos(va_i-va_j)) \
        #     - g_ij*(vm_i*vm_j*torch.sin(va_i-va_j))

        ###### another mine ######
        Pji = g_ij*(e_i*e_j-e_i**2+f_i*f_j-f_i**2) + b_ij*(f_i*e_j-e_i*f_j)
        Qji = g_ij*(f_i*e_j-e_i*f_j) + b_ij*(-e_i*e_j+e_i**2-f_i*f_j+f_i**2)

        # --- DEBUG ---
        # self._dPQ = torch.cat([Pji, Qji], dim=-1) # (num_edges, 2)
        # --- DEBUG ---

        return torch.cat([Pji, Qji], dim=-1) # (num_edges, 2)