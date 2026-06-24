import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=240, num_layers=3, actv=nn.Tanh):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(actv())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(actv())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HNN(nn.Module):
    def __init__(self, input_dim=10*3*2, hidden_dim=240, num_layers=3, actv=nn.Tanh):
        super().__init__()
        # 10 planets x 3 coords x 2 physical quantities (q,p) = 60 dim input
        self.net = MLP(input_dim, hidden_dim, num_layers, actv)

        self.n = input_dim // 2
        
        J_matrix = self.permutation_tensor(input_dim)
        self.register_buffer('J', J_matrix) # register J_matrix as buffer
       

    def forward(self, x):
        return self.net(x)
    
    def permutation_tensor(self,n):
        M = torch.eye(n)
        M = torch.cat([M[n//2:], -M[:n//2]])
        return M
        
    def get_grad(self, q, p, create_graph: bool = False):
        q_ = q.detach().requires_grad_(True)
        p_ = p.detach().requires_grad_(True)
        x = torch.cat([q_, p_], dim=-1)
        H = self.forward(x).sum()

        dH = torch.autograd.grad(
            H,
            [q_, p_],
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False
        )

        dH_dq, dH_dp = dH[0], dH[1]

        if not create_graph:
            return dH_dq.detach(), dH_dp.detach()
        else:
            return dH_dq, dH_dp

    def time_derivative(self, q, p, create_graph: bool = False):
        dHdq, dHdp = self.get_grad(q, p, create_graph)
        dx = torch.cat([dHdq, dHdp], dim=-1)  
        output = dx @ self.J.t()
   
        return output[:, :self.n], output[:, self.n:] # dq/dt, dp/dt