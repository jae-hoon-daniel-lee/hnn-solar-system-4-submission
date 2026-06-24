import numpy as np
import torch
from torch.utils.data import Dataset

class SolarSystemDataset(Dataset):
    def __init__(self, pos_path, mom_path, dpos_path, dmom_path, mode='train', train_ratio=0.8, stats=None):
        q_raw = np.load(pos_path)['q'].reshape(-1, 30)
        p_raw = np.load(mom_path)['p'].reshape(-1, 30)
        dq_raw = np.load(dpos_path)['dq'].reshape(-1, 30)
        dp_raw = np.load(dmom_path)['dp'].reshape(-1, 30)
        
        split_idx = int(len(q_raw) * train_ratio)
      
        if mode == 'train':
            self.q_max = np.abs(q_raw).max().astype('float64')
            self.p_max = np.abs(p_raw).max().astype('float64')
            self.dq_max = np.abs(dq_raw).max().astype('float64')
            self.dp_max = np.abs(dp_raw).max().astype('float64')
        else:
            self.q_max = stats['q_max']
            self.p_max = stats['p_max']
            self.dq_max = stats['dq_max']
            self.dp_max = stats['dp_max']

        q_scaled = q_raw / self.q_max
        p_scaled = p_raw / self.p_max
        dq_scaled = dq_raw / self.dq_max
        dp_scaled = dp_raw / self.dp_max

        if mode == 'train':
            self.q = torch.tensor(q_scaled[:split_idx], dtype=torch.float64)
            self.p = torch.tensor(p_scaled[:split_idx], dtype=torch.float64)
            self.dq = torch.tensor(dq_scaled[:split_idx], dtype=torch.float64)
            self.dp = torch.tensor(dp_scaled[:split_idx], dtype=torch.float64)
        else:
            self.q = torch.tensor(q_scaled[split_idx:], dtype=torch.float64)
            self.p = torch.tensor(p_scaled[split_idx:], dtype=torch.float64)
            self.dq = torch.tensor(dq_scaled[split_idx:], dtype=torch.float64)
            self.dp = torch.tensor(dp_scaled[split_idx:], dtype=torch.float64)

        print(f'data size in mode: {mode}')
        print('q:', self.q.shape)
        print('p:', self.p.shape)
        print('dq:', self.dq.shape)
        print('dp:', self.dp.shape)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, idx):
        # Input at t: (q, p), Target at t: (dq/dt, dp/dt) 
        return self.q[idx], self.p[idx], self.dq[idx], self.dp[idx]
    
    def get_stats(self):
        return {
            'q_max': self.q_max,
            'p_max': self.p_max,
            'dq_max': self.dq_max,
            'dp_max': self.dp_max
        }

