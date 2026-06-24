
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import sys
import joblib

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
  
nn_orbit_files = ['mltsc-baseline-hnn(3)(240)_predicted_orbits.npz',
                  'mltsc-v2-baseline-hnn(3)(240)_predicted_orbits.npz',
                  'unisc-baseline-hnn(3)(240)_predicted_orbits.npz']


# Ground Truth orbit file paths
q_path = '../data_ode/q.npz'
p_path = '../data_ode/p.npz'

# Ground Truth orbit loading
from utils_orbit import ground_truth_orbits

dt = 0.0005 # 0.0005 Time Units (years) per step
start_idx = 1600000 # at time = 1_600_000 * dt=0.0005 --> 800 Time Units (years)
eval_steps = [100000] # 50 years = 100_000 steps with dt=0.0005
gt_q, gt_p = ground_truth_orbits(q_path, p_path, start_idx, eval_steps[-1])

q_trues = []
p_trues = []
q_true, p_true = ground_truth_orbits(q_path, p_path, start_idx, eval_steps[-1])
q_trues.append(q_true)
p_trues.append(p_true)

q_preds = []
p_preds = []

for i in range(len(nn_orbit_files)):
    print(f"nn_orbit_file[{i}]:", nn_orbit_files[i])
    nn_orbit_path = f"{save_dir_path}/" + nn_orbit_files[i]
    with np.load(nn_orbit_path, allow_pickle=False) as data:
        print("keys in .npz:", data.files)
        if 'q' in data.files:
            q_pred = data['q']
        else:
            q_pred = data[data.files[0]]
        q_preds.append(q_pred)
        if 'p' in data.files:
            p_pred = data['p']
        else:
            p_pred = data[data.files[1]] if len(data.files) > 1 else None
        p_preds.append(p_pred)

        print(f"q_pred[{model_names[i]}] type/shape:", 
              type(q_pred), getattr(q_pred, 'shape', None))
        print(f"p_pred[{model_names[i]}] type/shape:", 
              type(p_pred), getattr(p_pred, 'shape', None))


from utils_energy import calculate_gt_energy_terms
from utils_energy import calculate_nn_energy_terms

gt_H_gt_x_list = []
gt_H_nn_x_list = []
gt_T_gt_x_list = []
gt_T_nn_x_list = []
gt_V_gt_x_list = []
gt_V_nn_x_list = []

from utils_chkpt import load_checkpoint

gt_T_gt_x, gt_V_gt_x, gt_H_gt_x = calculate_gt_energy_terms(q_true, p_true)

gt_H_gt_x_list.append(gt_H_gt_x)
gt_T_gt_x_list.append(gt_T_gt_x)
gt_V_gt_x_list.append(gt_V_gt_x)    

for i in range(len(model_names)):
    model_name = model_names[i]
    print(f"Processing model: {model_name}_ep{at_epocs[i]}")
    
    chkpt_dir = checkpoint_dir_paths[i]
    epoch = at_epocs[i]
    total_epochs = total_epocs[i]
    lr = lrs[i]
    pat = pats[i]
    scstat = scstats[i]

    # Load the model checkpoint
    model = HNN(input_dim=60).to(device)
    model.double() # double precision
    checkpoint_path = chkpt_dir + "/" \
                    + f"{model_name}_ep_{epoch}_of_{total_epochs}" \
                    + f"_bs_8192_lr_{lr:.0e}_pat_{pat}.pth"
    _, _ = load_checkpoint(model, optimizer=None, file_path=checkpoint_path)

    # Load scaling statistics
    scstat_path = chkpt_dir + "/" + scstat
    scaling_stats = joblib.load(scstat_path)

    q_max = scaling_stats['q_max']
    p_max = scaling_stats['p_max']
    dq_max = scaling_stats['dq_max']
    dp_max = scaling_stats['dp_max']

    q_pred = q_preds[i]
    p_pred = p_preds[i]

    gt_T_nn_x, gt_V_nn_x, gt_H_nn_x = calculate_gt_energy_terms(q_pred, p_pred)

    gt_H_nn_x_list.append(gt_H_nn_x)
    gt_T_nn_x_list.append(gt_T_nn_x)
    gt_V_nn_x_list.append(gt_V_nn_x)    

save_path = f"{save_dir_path}/gt_energy_gt_x_terms_50_years.npz"
np.savez_compressed(save_path, 
                    T=gt_T_gt_x_list[0],
                    V=gt_V_gt_x_list[0],     
                    H=gt_H_gt_x_list[0])

for i in range(len(model_names)):
    model_name = model_names[i]
    save_path = f"{save_dir_path}/{model_name}_gt_energy_nn_x_terms_50_years.npz"
    np.savez_compressed(save_path,
                        T=gt_T_nn_x_list[i],
                        V=gt_V_nn_x_list[i],        
                        H=gt_H_nn_x_list[i])

