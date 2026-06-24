
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

block_plausible_hnn_dir_name = 'experiment_block-plausible-hnn_lite'

block_plausible_hnn_dir_path = os.path.join(os.path.dirname(os.getcwd()), block_plausible_hnn_dir_name)
print(f"Adding {block_plausible_hnn_dir_path} to sys.path for module imports.")

if block_plausible_hnn_dir_path not in sys.path:
    sys.path.append(block_plausible_hnn_dir_path)

save_dir_path = "./harvests"
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path, exist_ok=True)


model_prefix='opt-att-'
model_names=['blk-p-hnn(4)(7)-bs200']
at_epocs=[36]
facs = [7]

checkpoint_dir_path = f"{block_plausible_hnn_dir_path}/checkpoints/multi-scale-v2"

partIdx = 1
partIds = [1, 2]
startIds = [320000, 360000]

from nn_model_block_plausible import BlockPlausibleHNN
from utils_chkpt import load_checkpoint
from utils_orbit import scipy_predict_qp_model_orbits_and_save

def get_eval_objs(model_names, at_epocs, facs):
    eval_objs = []

    for model_name, epoch, fac in zip(model_names, at_epocs, facs):
        print(f"Processing model: {model_name}_ep{epoch}")
    
        # Load the model checkpoint
        model = BlockPlausibleHNN(m=fac, n_layers=4).to(device)
        model.double()
        checkpoint_path = checkpoint_dir_path \
                        + f"/{model_prefix}{model_name}_ep_{epoch}_of_60" \
                        + "_lr_1e-04_pat_3.pth"
        print(f'checkpoint path: {checkpoint_path}')

        _, _ = load_checkpoint(model, optimizer=None, file_path=checkpoint_path)
    
        # Load scaling statistics
        scstat_path = f"{checkpoint_dir_path}/ds_mlt_scalev.pkl"
        scaling_stats = joblib.load(scstat_path)
    
        q_max = scaling_stats['q_max']
        p_max = scaling_stats['p_max']
        dq_max = scaling_stats['dq_max']
        dp_max = scaling_stats['dp_max']

        eval_objs.append({'model': model, 'q_max': q_max, 'p_max': p_max, 'dq_max': dq_max, 'dp_max': dp_max})

        print(model)
    
    return eval_objs

def gather_orbits():
    dt = 0.01 # 0.01 Time Units (years) per step
    part_idx = partIds[partIdx]
    start_idx = startIds[part_idx-1] # at time = 320000 * dt=0.01 --> 3200 Time Units (years)

    eval_steps = [40000] # 400 years = 40000 steps with dt=0.01
    
    # Ground Truth orbit file paths
    q_path = '../data_ode/q_lite.npz'
    p_path = '../data_ode/p_lite.npz'

    # Ground Truth orbit loading
    from utils_orbit import ground_truth_orbits
    gt_q, gt_p = ground_truth_orbits(q_path, p_path, start_idx, eval_steps[-1])
    
    # Model prediction (fed with the 1st frame of Ground Truth)
    q_init_val = gt_q[0:1]
    p_init_val = gt_p[0:1]
    
    eval_objs = get_eval_objs(model_names, at_epocs, facs)
    
    # Predict orbits and save results
    for i, eval_obj in enumerate(eval_objs):
        model = eval_obj['model']
        q_max = eval_obj['q_max']
        p_max = eval_obj['p_max']
        dq_max = eval_obj['dq_max']
        dp_max = eval_obj['dp_max']

        save_file_name = f"{model_names[i]}" \
                       + f"_ep{at_epocs[i]}_part{part_idx}" \
                       + "_predicted_orbits.npz"
        scipy_predict_qp_model_orbits_and_save(model, device, 
                                           q_init_val, p_init_val, eval_steps[-1], dt, 
                                           q_max, p_max, dq_max, dp_max,
                                           save_dir=save_dir_path, 
                                           save_file_name=save_file_name)
        
if __name__ == "__main__":
    gather_orbits()