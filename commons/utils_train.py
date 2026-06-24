
import os
import sys
import torch
import random
import numpy as np

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ Seed set to {seed} for reproducibility")

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def train_one_epoch(model, optimizer, train_loader, device, grad_max_norm=1.0, grad_clipping=False):
    model.train()
    train_epoch_loss = 0
        
    for batch_idx, (q_t, p_t, dq_t, dp_t) in enumerate(train_loader):
        q_t, p_t = q_t.to(device), p_t.to(device)
        dq_t, dp_t = dq_t.to(device), dp_t.to(device)

        with torch.enable_grad():
            dq_m, dp_m = model.time_derivative(q_t, p_t, create_graph=True)
            
        optimizer.zero_grad()

        loss = torch.nn.functional.mse_loss(dq_t, dq_m) + \
               torch.nn.functional.mse_loss(dp_t, dp_m)
            
        loss.backward()

        if grad_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_max_norm) 
        
        optimizer.step()
        train_epoch_loss += loss.item()

    avg_train_loss = train_epoch_loss / len(train_loader)

    return avg_train_loss


def test_one_epoch(model, test_loader, device):
    model.eval()
    test_epoch_loss = 0
        
    for batch_idx, (q_t, p_t, dq_t, dp_t) in enumerate(test_loader):
        q_t, p_t = q_t.to(device), p_t.to(device)
        dq_t, dp_t = dq_t.to(device), dp_t.to(device)

        with torch.enable_grad():
            dq_m, dp_m = model.time_derivative(q_t, p_t, create_graph=False)

        t_loss = torch.nn.functional.mse_loss(dq_t, dq_m) + \
                 torch.nn.functional.mse_loss(dp_t, dp_m)
        test_epoch_loss += t_loss.item()
        
    avg_test_loss = test_epoch_loss / len(test_loader)

    return avg_test_loss
