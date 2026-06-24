
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

save_dir_path = "./harvests"
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path, exist_ok=True)

from nn_model_plausible import PlausibleHNN

model_names=['p-hnn(4)(10)']
at_epocs=[16]
facs = [10]

checkpoint_dir_path="../experiment_plausible-hnn_lite/checkpoints/multi-scale-v2"

nn_orbit_files = ['p-hnn(4)(10)_part1_predicted_orbits.npz']

# Ground Truth orbit file paths
q_path = '../data_ode/q_lite.npz'
p_path = '../data_ode/p_lite.npz'

# Ground Truth orbit loading
from utils_orbit import ground_truth_orbits

dt = 0.01 # 0.01 Time Units (years) per step
start_idx = 320000 # at time = 320000 * dt=0.01 --> 3200 Time Units (years)
eval_steps = [40000] # 400 years = 40000 steps with dt=0.01
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
    nn_orbit_path = save_dir_path + "/" + nn_orbit_files[i]
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
nn_H_gt_x_list = []
nn_H_nn_x_list = []
nn_T_gt_x_list = []
nn_T_nn_x_list = []
nn_V_gt_x_list = []
nn_V_nn_x_list = []

from utils_chkpt import load_checkpoint

gt_T_gt_x, gt_V_gt_x, gt_H_gt_x = calculate_gt_energy_terms(q_true, p_true)

gt_H_gt_x_list.append(gt_H_gt_x)
gt_T_gt_x_list.append(gt_T_gt_x)
gt_V_gt_x_list.append(gt_V_gt_x)    

for i in range(len(model_names)):
    model_name = model_names[i]
    print(f"Processing model: {model_name}_ep{at_epocs[i]}")
    
    # Load the model checkpoint
    model = PlausibleHNN(fac=facs[i], num_layers=4).to(device)
    model.double() # double precision
    checkpoint_path = f"{checkpoint_dir_path}/{model_name}" \
                    + f"_ep_{at_epocs[i]}_of_30_bs_200" \
                    + f"_lr_1e-04_pat_3.pth"
    print(f'checkpoint path: {checkpoint_path}')

    _, _ = load_checkpoint(model, optimizer=None, file_path=checkpoint_path)

    # Load scaling statistics
    scstat_path = f"{checkpoint_dir_path}/ds_mlt_scalev.pkl"
    scaling_stats = joblib.load(scstat_path)

    q_max = scaling_stats['q_max']
    p_max = scaling_stats['p_max']
    dq_max = scaling_stats['dq_max']
    dp_max = scaling_stats['dp_max']

    q_pred = q_preds[i]
    p_pred = p_preds[i]

    gt_T_nn_x, gt_V_nn_x, gt_H_nn_x = calculate_gt_energy_terms(q_pred, p_pred)
    nn_T_gt_x, nn_V_gt_x, nn_H_gt_x = calculate_nn_energy_terms(model, device, 
                                                                q_true, p_true, 
                                                                q_max, p_max, qp_cat=False)
    nn_T_nn_x, nn_V_nn_x, nn_H_nn_x = calculate_nn_energy_terms(model, device, 
                                                                q_pred, p_pred, 
                                                                q_max, p_max, qp_cat=False)
    gt_H_nn_x_list.append(gt_H_nn_x)
    gt_T_nn_x_list.append(gt_T_nn_x)
    gt_V_nn_x_list.append(gt_V_nn_x) 
    nn_H_gt_x_list.append(nn_H_gt_x)
    nn_T_gt_x_list.append(nn_T_gt_x)
    nn_V_gt_x_list.append(nn_V_gt_x)
    nn_H_nn_x_list.append(nn_H_nn_x)
    nn_T_nn_x_list.append(nn_T_nn_x)
    nn_V_nn_x_list.append(nn_V_nn_x)   

save_path = f"{save_dir_path}/gt_energy_gt_x_terms_part1_400_years.npz"
np.savez_compressed(save_path, 
                    T=gt_T_gt_x_list[0],
                    V=gt_V_gt_x_list[0],     
                    H=gt_H_gt_x_list[0])

for i in range(len(model_names)):
    model_name = model_names[i]
    save_path = f"{save_dir_path}/{model_name}_gt_energy_nn_x_terms_part1_400_years.npz"
    np.savez_compressed(save_path,
                        T=gt_T_nn_x_list[i],
                        V=gt_V_nn_x_list[i],        
                        H=gt_H_nn_x_list[i])
    
for i in range(len(model_names)):
    model_name = model_names[i]
    save_path = f"{save_dir_path}/{model_name}_nn_energy_gt_x_terms_part1_400_years.npz"
    np.savez_compressed(save_path,
                        T=nn_T_gt_x_list[i],
                        V=nn_V_gt_x_list[i],        
                        H=nn_H_gt_x_list[i])

for i in range(len(model_names)):
    model_name = model_names[i]
    save_path = f"{save_dir_path}/{model_name}_nn_energy_nn_x_terms_part1_400_years.npz"
    np.savez_compressed(save_path,
                        T=nn_T_nn_x_list[i],
                        V=nn_V_nn_x_list[i],        
                        H=nn_H_nn_x_list[i])
