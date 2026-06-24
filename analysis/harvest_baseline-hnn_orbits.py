
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import gc
gc.collect()

common_dir_name = 'commons' 
common_dir_path = os.path.join(os.path.dirname(os.getcwd()), common_dir_name)
print(f"Adding {common_dir_path} to sys.path for module imports.")

if common_dir_path not in sys.path:
    sys.path.append(common_dir_path)

data_dir_name = 'data_ode' 
data_dir_path = os.path.join(os.path.dirname(os.getcwd()), data_dir_name)
print(f"Adding {data_dir_path} to sys.path for module imports.")

if data_dir_path not in sys.path:
    sys.path.append(data_dir_path)

baseline_hnn_dir_name = 'experiment_baseline-hnn'

baseline_hnn_dir_path = os.path.join(os.path.dirname(os.getcwd()), baseline_hnn_dir_name)
print(f"Adding {baseline_hnn_dir_path} to sys.path for module imports.")

if baseline_hnn_dir_path not in sys.path:
    sys.path.append(baseline_hnn_dir_path)

save_dir_path = "./harvests"
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path, exist_ok=True)

model_names=['mltsc-baseline-hnn(3)(240)', 
             'mltsc-v2-baseline-hnn(3)(240)', 
             'unisc-baseline-hnn(3)(240)']
at_epocs=[500, 19, 19934]
total_epocs=[500, 150, 20000]
lrs=[5e-03, 5e-03, 1e-2]
pats=[2, 2, 3]
scstats=["ds_mlt_scalev.pkl",
         "ds_mlt_scalev.pkl", 
         "ds_uni_scalev.pkl"]

checkpoint_dir_paths = [f"{baseline_hnn_dir_path}/checkpoints/multi-scale",
                        f"{baseline_hnn_dir_path}/checkpoints/multi-scale-v2",
                        f"{baseline_hnn_dir_path}/checkpoints/uni-scale"]

from nn_model_baseline import HNN

from utils_chkpt import load_checkpoint
from utils_orbit import scipy_predict_qp_model_orbits_and_save

def get_eval_objs(checkpoint_dir_paths,
                  model_names, 
                  at_epocs, total_epocs, 
                  lrs, pats, 
                  scstats):
    eval_objs = []

    for chkpt_dir, model_name, epoch, total_epochs, lr, pat, scstat \
    in zip(checkpoint_dir_paths, model_names, at_epocs, total_epocs, lrs, pats, scstats):
        print(f"Processing model: {model_name}_ep{epoch}")
    
        # Load the model checkpoint
        model = HNN(hidden_dim=240, num_layers=3).to(device)
        model.double() # double precision
        checkpoint_path = chkpt_dir + "/" \
                        + f"{model_name}_ep_{epoch}_of_{total_epochs}" \
                        + f"_bs_8192_lr_{lr:.0e}_pat_{pat}.pth"
        print(f'checkpoint path: {checkpoint_path}')

        _, _ = load_checkpoint(model, optimizer=None, file_path=checkpoint_path)
    
        # Load scaling statistics
        scstat_path = f"{chkpt_dir}/{scstat}"
        scaling_stats = joblib.load(scstat_path)
    
        q_max = scaling_stats['q_max']
        p_max = scaling_stats['p_max']
        dq_max = scaling_stats['dq_max']
        dp_max = scaling_stats['dp_max']

        eval_objs.append({'model': model, 'q_max': q_max, 'p_max': p_max, 'dq_max': dq_max, 'dp_max': dp_max})

        print(model)
    
    return eval_objs

def gather_orbits():
    dt = 0.0005 # 0.0005 Time Units (years) per step
    start_idx = 1600000 # at time = 1_600_000 * dt=0.0005 --> 800 Time Units (years)

    eval_steps = [100000] # 50 years = 100_000 steps with dt=0.0005
    
    # Ground Truth orbit file paths
    q_path = '../data_ode/q.npz'
    p_path = '../data_ode/p.npz'

    # Ground Truth orbit loading
    from utils_orbit import ground_truth_orbits
    gt_q, gt_p = ground_truth_orbits(q_path, p_path, start_idx, eval_steps[-1])
    
    # Model prediction (fed with the 1st frame of Ground Truth)
    q_init_val = gt_q[0:1]
    p_init_val = gt_p[0:1]
    
    eval_objs = get_eval_objs(checkpoint_dir_paths,
                              model_names, 
                              at_epocs, total_epocs, 
                              lrs, pats, 
                              scstats)
    
    # Predict orbits and save results
    for i, eval_obj in enumerate(eval_objs):
        model = eval_obj['model']
        q_max = eval_obj['q_max']
        p_max = eval_obj['p_max']
        dq_max = eval_obj['dq_max']
        dp_max = eval_obj['dp_max']

        save_file_name = f"{model_names[i]}_predicted_orbits.npz"
        scipy_predict_qp_model_orbits_and_save(model, device, 
                                           q_init_val, p_init_val, eval_steps[-1], dt, 
                                           q_max, p_max, dq_max, dp_max,
                                           save_dir=save_dir_path,
                                           save_file_name=save_file_name)
        
if __name__ == "__main__":
    gather_orbits()