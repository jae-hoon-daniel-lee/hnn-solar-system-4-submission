
import torch
import torch.nn as nn
import torch.nn.functional as F

class EvenActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # f(x) = ln (1 + x tanh(x)): even
        return torch.log1p(x * torch.tanh(x))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=240, num_layers=3, actv=nn.Tanh):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(actv())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(actv())
        layers.append(nn.Linear(hidden_dim, 1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PlausibleHNN(nn.Module):
    def __init__(self, num_bodies=10, dims=3, fac=2, num_layers=3):
        super().__init__()
        self.n = num_bodies * dims
        self.n_pairs = num_bodies * (num_bodies - 1) // 2 # 45
        self.hidden_dim = self.n_pairs * dims * fac  # 135 x fac
        # 135 x fac(2) = 270, 135 x fac(4) = 540, 
        # 125 x fac(6) = 810, 135 x fac(8) = 1080
        # 135 x fac(10) = 1350, 135 x fac(12) = 1620
        
        # 1. T-net
        self.w = nn.Parameter(torch.ones(self.n))
        
        # 2. V-net
        self.v_mlp = MLP(
            input_dim=self.n, 
            hidden_dim=self.hidden_dim, 
            num_layers=num_layers, 
            actv=EvenActivation
        )

        self.T, self.V, self.H = None, None, None

    def get_T(self, p_hat):
        # T(p) = sum(w * p_hat^2)
        return torch.sum(self.w * (p_hat**2), dim=1, keepdim=True)

    def get_V(self, q_hat):
        # V(q) = MLP(q_hat)
        return self.v_mlp(q_hat)

    def forward(self, q_hat, p_hat):
        self.T = self.get_T(p_hat)
        self.V = self.get_V(q_hat)
        self.H = self.T + self.V
        return self.H
    
    def energy_terms(self):
        return self.T, self.V
    
    def get_grad(self, q_hat, p_hat, create_graph: bool = False):
        qh = q_hat.detach().requires_grad_(True)
        ph = p_hat.detach().requires_grad_(True)
        
        H = self.forward(qh, ph)
        H_sum = H.sum()

        grads = torch.autograd.grad(
            H_sum, [qh, ph],
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False
        )
        return grads[0], grads[1] # dHdq, dHdp

    def time_derivative(self, q_hat, p_hat, create_graph: bool = False):
        dHdq, dHdp = self.get_grad(q_hat, p_hat, create_graph)
        return dHdp, -dHdq # dq/dt, dp/dt
