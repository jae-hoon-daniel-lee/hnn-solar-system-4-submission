
import numpy as np
import os
import sys

common_dir_name = 'commons' 
common_dir_path = os.path.join(os.path.dirname(os.getcwd()), common_dir_name)
if common_dir_path not in sys.path:
    sys.path.append(common_dir_path)

data_dir_name = 'data_ode'
data_dir_path = os.path.join(os.path.dirname(os.getcwd()), data_dir_name)
if data_dir_path not in sys.path:
    sys.path.append(data_dir_path)

analysis_tool_path = './analysis_tools'
if analysis_tool_path not in sys.path:
    sys.path.append(analysis_tool_path)

dt = 0.01

eval_partitions_years=[{'from': 3200, 'to': 3600}, 
                 {'from': 3600, 'to': 4000 }]
eval_partitions_steps = [{'from': 320000, 'to': 360000},
                 {'from': 360000, 'to': 400000 }]

data4plot_dir="./harvests/"

epoc = 36

nn_orbit_files = [f"blk-p-hnn(4)(7)-bs200_ep{epoc}_part1_predicted_orbits.npz", 
                  f"blk-p-hnn(4)(7)-bs200_ep{epoc}_part2_predicted_orbits.npz"]

model_names = ["BLK-P-HNN(l=4,m=7)"]

save_model_name = f"blk-p-hnn-l4-m7-ep{epoc}"

save_fig_dir = "./plots"
img_save = True

max_pred_years = 400

current_pred_years = 400 # 350 # 300 # 250 # 200 # 100 # 50

pred_years = current_pred_years
pred_steps = int(pred_years / dt)



import importlib
import inspect

m = importlib.import_module("utils_plot_orbit_per_body")
print("module file:", getattr(m, "__file__", None))

from utils_plot_orbit_per_body import compare_plot2d_orbits_per_planet
from utils_plot_orbit_per_body import compare_plot3d_orbits_per_planet

from utils_orbit import ground_truth_orbits

q_paths = []
p_paths = []

q_trues = []
p_trues = []

q_path = f"{data_dir_path}/q_lite.npz"
p_path = f"{data_dir_path}/p_lite.npz"

for i in range(len(eval_partitions_years)):
    start_idx = eval_partitions_steps[i]['from']
    end_idx = eval_partitions_steps[i]['to']
    max_stp = end_idx - start_idx
    print(f"start_idx[{i}]:", start_idx, ", max_step:", max_stp)

    q_true, p_true = ground_truth_orbits(q_path, p_path, start_idx, max_stp)
    q_trues.append(q_true)
    p_trues.append(p_true)

    print(f"q_true[partition {i}] shape:", q_true.shape)
    print(f"p_true[partition {i}] shape:", p_true.shape)

q_preds = []
p_preds = []

for i in range(len(nn_orbit_files)):
    nn_orbit_path = data4plot_dir + nn_orbit_files[i]
    print(f"nn_orbit_path[{i}]:", nn_orbit_path)
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

        print(f"q_pred[partition {i}] type/shape:", 
              type(q_pred), getattr(q_pred, 'shape', None))
        print(f"p_pred[partition {i}] type/shape:", 
              type(p_pred), getattr(p_pred, 'shape', None))
        
planets = ["Sun",
           "Mercury",
           "Venus",
           "Earth",
           "Mars",
           "Jupiter",
           "Saturn",
           "Uranus",
           "Neptune",
           "Pluto"]

for i in range(len(eval_partitions_years)):
    print("q_pred shape before reshaping:", q_preds[i].shape)
    print("q_true shape before reshaping:", q_trues[i].shape)
    print("p_pred shape before reshaping:", p_preds[i].shape)
    print("p_true shape before reshaping:", p_trues[i].shape)
    
    q_preds[i] = q_preds[i].reshape(q_preds[i].shape[0], q_preds[i].shape[1]//3, 3) \
    if q_preds[i].ndim == 2 else q_preds[i]

    q_trues[i] = q_trues[i].reshape(q_trues[i].shape[0], q_trues[i].shape[1]//3, 3) \
    if q_trues[i].ndim == 2 else q_trues[i]

    p_preds[i] = p_preds[i].reshape(p_preds[i].shape[0], p_preds[i].shape[1]//3, 3) \
    if p_preds[i].ndim == 2 else p_preds[i]

    p_trues[i] = p_trues[i].reshape(p_trues[i].shape[0], p_trues[i].shape[1]//3, 3) \
    if p_trues[i].ndim == 2 else p_trues[i]

print("After reshaping:")
for i in range(len(eval_partitions_years)):
     print(f"Partition {i}:")
     print(f"q_pred[partition {i}] shape:", q_preds[i].shape)
     print(f"q_true[partition {i}] shape:", q_trues[i].shape)
     print(f"p_pred[partition {i}] shape:", p_preds[i].shape)
     print(f"p_true[partition {i}] shape:", p_trues[i].shape)

for i in range(len(eval_partitions_years)):
    q_pred = q_preds[i]
    q_true = q_trues[i]
    p_pred = p_preds[i]
    p_true = p_trues[i]
    diff_q = np.linalg.norm(q_pred - q_true, axis=-1)
    diff_p = np.linalg.norm(p_pred - p_true, axis=-1)

    diff_qx = np.linalg.norm(q_pred[:, :, 0] - q_true[:, :, 0], axis=-1)
    diff_qy = np.linalg.norm(q_pred[:, :, 1] - q_true[:, :, 1], axis=-1)
    diff_qz = np.linalg.norm(q_pred[:, :, 2] - q_true[:, :, 2], axis=-1)

    for j in range(10):
        diff_qx_j = np.linalg.norm(q_pred[:, j, 0] - q_true[:, j, 0])
        diff_qy_j = np.linalg.norm(q_pred[:, j, 1] - q_true[:, j, 1])
        diff_qz_j = np.linalg.norm(q_pred[:, j, 2] - q_true[:, j, 2])
        diff_qx_j_max = np.max(np.abs(q_pred[:, j, 0] - q_true[:, j, 0]))
        diff_qy_j_max = np.max(np.abs(q_pred[:, j, 1] - q_true[:, j, 1]))
        diff_qz_j_max = np.max(np.abs(q_pred[:, j, 2] - q_true[:, j, 2]))

        print(f"{planets[j]}: diff_qx[partition {i}]={diff_qx_j:.3e}") 
        print(f"{planets[j]}: diff_qy[partition {i}]={diff_qy_j:.3e}")
        print(f"{planets[j]}: diff_qz[partition {i}]={diff_qz_j:.3e}")

for i in range(len(eval_partitions_years)):
    q_pred = q_preds[i][0:pred_steps]
    q_true = q_trues[i][0:pred_steps]
    
    compare_plot3d_orbits_per_planet(q_pred, q_true, pred_years=pred_years, 
                   model_name=save_model_name,
                   num_partitions = len(eval_partitions_steps),
                   partition_idx = i, 
                   save_dir=f"./{save_fig_dir}", 
                   img_save=img_save)
    
for i in range(len(eval_partitions_years)):
    q_pred = q_preds[i][0:pred_steps]
    q_true = q_trues[i][0:pred_steps]
    
    compare_plot2d_orbits_per_planet(q_pred, q_true, pred_years=pred_years,
                   model_name=save_model_name,
                   num_partitions = len(eval_partitions_years),
                   partition_idx = i, 
                   save_dir=f"./{save_fig_dir}", 
                   img_save=img_save)
