
import torch
import torch.nn as nn
import itertools

from nn_model_plausible import EvenActivation

class PairWiseInputLinearAdaptor(nn.Module):
    def __init__(self, num_bodies=10, dims=3, m_fac=4*8):
        super().__init__()
        self.num_bodies = num_bodies
        self.dims = dims
        self.m = m_fac
        self.pairs = list(itertools.combinations(range(num_bodies), 2))
        self.n_pairs = len(self.pairs)
        
        indices = []
        for i, j in self.pairs:
            indices.extend([i*3, i*3+1, i*3+2, j*3, j*3+1, j*3+2])
        
        self.register_buffer('pair_indices', torch.tensor(indices, dtype=torch.long))
        
        # [45, m, 6]
        self.weight = nn.Parameter(torch.empty(self.n_pairs, self.m, 2 * dims))
        self.bias = nn.Parameter(torch.Tensor(self.n_pairs, self.m))

        nn.init.xavier_uniform_(self.weight, gain=1.0)
        nn.init.zeros_(self.bias)

    def forward(self, x):   
        # x: [B, 30] -> [B, 45, 6]
        batch_size = x.shape[0]
        pair_x = x.index_select(1, self.pair_indices).view(batch_size, self.n_pairs, 6)
    
        # [B, 45, 6] -> [45, B, 6] -> [45, 6, B]
        x_bmm = pair_x.transpose(0, 1).transpose(1, 2) 
        # [45, m, 6] @ [45, 6, B] -> [45, m, B]
        # unsqueeze(-1): [45, m] -> [45, m, 1]
        y = torch.bmm(self.weight, x_bmm) + self.bias.unsqueeze(-1)
    
    
        # [45, m, B] -> [B, 45, m]
        # [45, m, B].transpose(1, 2) => [45, B, m]
        # [45, B, m].transpose(0, 1) => [B, 45, m]
        return y.transpose(1, 2).transpose(0, 1)
    
class PairWiseHiddenBlockLinear(nn.Module):
    def __init__(self, n_pairs, m_fac=4*8):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(n_pairs, m_fac, m_fac))
        self.bias = nn.Parameter(torch.empty(n_pairs, m_fac))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [B, 45, m]
        # [45, m, B]
        x_bmm = x.transpose(0, 1).transpose(1, 2)
    
        # unsqueeze(-1): [45, m] -> [45, m, 1]
        y = torch.bmm(self.weight, x_bmm) + self.bias.unsqueeze(-1)
    
        # [B, 45, m]
        return y.transpose(1, 2).transpose(0, 1)
    
class PairWiseLastLayer(nn.Module):
    def __init__(self, n_pairs, m=4*8):
        super().__init__()
        # [45, 1, m]
        # Final energy pooling weight: [45, 1, m]
        self.W = nn.Parameter(torch.empty(n_pairs, 1, m))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x):
        # x: [B, 45, m] -> [45, m, B]
        x_bmm = x.transpose(0, 1).transpose(1, 2)
        
        # bmm: [45, 1, m] @ [45, m, B] -> [45, 1, B]
        v_ij_bmm = torch.bmm(self.W, x_bmm)
        
        # [45, 1, B] -> [B, 45]
        # v_ij_bmm.squeeze(1) => [45, B]
        # .transpose(0, 1) => [B, 45]
        return v_ij_bmm.squeeze(1).transpose(0, 1)

balancing_options = ["default", "attention"]
    
class BlockPlausibleHNN(nn.Module):
    def __init__(self, n_bodies=10, dims=3, m=4*8, n_layers=3, option="attention"):
        super().__init__()
        self.n_pairs = n_bodies * (n_bodies - 1) // 2
        self.n = n_bodies * dims
        self.m = m
        
        # T-net
        self.w = nn.Parameter(torch.ones(n_bodies * dims) * 0.5)
        
        # V-net
        # 1. Adaptor: [B, 30] -> [B, 45, 6] (Simplified q_i, q_j extractor)
        # 2. Hidden1: [B, 45, m]
        # 3. Hidden2: [B, 45, m]
        # 4. Last: [B, 45] -> sum -> [B, 1]

        layers = []
        layers.append(PairWiseInputLinearAdaptor(n_bodies, dims, m)) # Custom Adaptor
        for i in range(n_layers - 2):
            layers.append(EvenActivation())
            layers.append(PairWiseHiddenBlockLinear(self.n_pairs, self.m))
        layers.append(EvenActivation())       
        layers.append(PairWiseLastLayer(self.n_pairs, self.m))
        self.v_mlp = nn.Sequential(*layers)

        if option == "attention":
            self.w_v = nn.Parameter(torch.ones(self.n_pairs))
            self.get_V = self._get_V_by_attention
        else:
            self.get_V = self._get_V_default

    def _get_V_default(self, q):
        # V-net
        # v_mlp's output shape: [batch, 45]
        v_ij = self.v_mlp(q) 
        
        # V = Σ V_ij
        # output shape: [batch, 1]
        return torch.sum(v_ij, dim=1, keepdim=True)
    
    def _get_V_by_attention(self, q):
        v_ij = self.v_mlp(q) # [Batch, 45]
        return torch.sum(self.w_v * v_ij, dim=1, keepdim=True)
    
    def get_T(self, p):
        return torch.sum(self.w * (p**2), dim=1, keepdim=True)
    
    def forward(self, q, p):
        self.V, self.T = self.get_V(q), self.get_T(p)
        return self.V + self.T

    def energy_terms(self):
        return self.T, self.V
    
    def get_grad(self, q, p, create_graph: bool = False):
        qh = q.detach().requires_grad_(True)
        ph = p.detach().requires_grad_(True)
        
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
