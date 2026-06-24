
import numpy as np
import os, sys
import matplotlib.pyplot as plt

analysis_tool_dir = './analysis_tools'
sys.path.append(analysis_tool_dir)

model_names = ['p-hnn-l4-m10']

data4plot_dir = './harvests/'
plot_dir = './plots'
plot_with_T_and_V = False

gt_energy_gt_x_filenames = ['gt_energy_gt_x_terms_part1_400_years.npz']
gt_energy_nn_x_filenames = ['p-hnn(4)(10)_gt_energy_nn_x_terms_part1_400_years.npz']
nn_energy_gt_x_filenames = ['p-hnn(4)(10)_nn_energy_gt_x_terms_part1_400_years.npz']
nn_energy_nn_x_filenames = ['p-hnn(4)(10)_nn_energy_nn_x_terms_part1_400_years.npz']

gt_Ts_gt_x, gt_Vs_gt_x, gt_Hs_gt_x = [], [], []
gt_Ts_nn_x, gt_Vs_nn_x, gt_Hs_nn_x = [], [], []
nn_Ts_gt_x, nn_Vs_gt_x, nn_Hs_gt_x = [], [], []
nn_Ts_nn_x, nn_Vs_nn_x, nn_Hs_nn_x = [], [], []

for i in range(len(gt_energy_gt_x_filenames)):
    gt_energy_gt_x_path=data4plot_dir + gt_energy_gt_x_filenames[i]
    with np.load(gt_energy_gt_x_path, allow_pickle=False) as data:
        print("keys in .npz:", data.files)
        if 'T' in data.files:
            gt_T = data['T']
        else:
            gt_T = data[data.files[0]]
        gt_Ts_gt_x.append(gt_T)
        if 'V' in data.files:
            gt_V = data['V']
        else:
            gt_V = data[data.files[1]] if len(data.files) > 1 else None
        gt_Vs_gt_x.append(gt_V)
        if 'H' in data.files:
            gt_H = data['H']
        else:
            gt_H = data[data.files[2]] if len(data.files) > 2 else None
        gt_Hs_gt_x.append(gt_H)

        print(f"gt_T_gt_x[{model_names[i]}] type/shape:", 
              type(gt_T), getattr(gt_T, 'shape', None))
        print(f"gt_V_gt_x[{model_names[i]}] type/shape:", 
              type(gt_V), getattr(gt_V, 'shape', None))
        print(f"gt_H_gt_x[{model_names[i]}] type/shape:", 
              type(gt_H), getattr(gt_H, 'shape', None))
        print('gt_T_gt_x sample values:', gt_T[:5] if gt_T is not None else None)
        print('gt_V_gt_x sample values:', gt_V[:5] if gt_V is not None else None)
        print('gt_H_gt_x sample values:', gt_H[:5] if gt_H is not None else None)

for i in range(len(gt_energy_nn_x_filenames)):
    gt_energy_nn_x_path=data4plot_dir + gt_energy_nn_x_filenames[i]
    print(f"gt_energy_nn_x_path[{i}]:", gt_energy_nn_x_path)
    with np.load(gt_energy_nn_x_path, allow_pickle=False) as data:
        print("keys in .npz:", data.files)
        if 'T' in data.files:
            gt_T = data['T']
        else:
            gt_T = data[data.files[0]]
        gt_Ts_nn_x.append(gt_T)
        if 'V' in data.files:
            gt_V = data['V']
        else:
            gt_V = data[data.files[1]] if len(data.files) > 1 else None
        gt_Vs_nn_x.append(gt_V)
        if 'H' in data.files:
            gt_H = data['H']
        else:
            gt_H = data[data.files[2]] if len(data.files) > 2 else None
        gt_Hs_nn_x.append(gt_H)
        pinx = i + 1
        print(f"gt_T_nn_x[partition {pinx}] type/shape:", 
              type(gt_T), getattr(gt_T, 'shape', None))
        print(f"gt_V_nn_x[partition {pinx}] type/shape:", 
              type(gt_V), getattr(gt_V, 'shape', None))
        print(f"gt_H_nn_x[partition {pinx}] type/shape:", 
              type(gt_H), getattr(gt_H, 'shape', None))
        print('gt_T_nn_x sample values:', gt_T[:5] if gt_T is not None else None)
        print('gt_V_nn_x sample values:', gt_V[:5] if gt_V is not None else None)
        print('gt_H_nn_x sample values:', gt_H[:5] if gt_H is not None else None)

for i in range(len(nn_energy_gt_x_filenames)):
    nn_energy_gt_x_path=data4plot_dir + nn_energy_gt_x_filenames[i]
    print(f"nn_energy_gt_x_path[{i}]:", nn_energy_gt_x_path)
    with np.load(nn_energy_gt_x_path, allow_pickle=False) as data:
        print("keys in .npz:", data.files)
        if 'T' in data.files:
            nn_T = data['T']
        else:
            nn_T = data[data.files[0]]
        nn_Ts_gt_x.append(nn_T)
        if 'V' in data.files:
            nn_V = data['V']
        else:
            nn_V = data[data.files[1]] if len(data.files) > 1 else None
        nn_Vs_gt_x.append(nn_V)
        if 'H' in data.files:
            nn_H = data['H']
        else:
            nn_H = data[data.files[2]] if len(data.files) > 2 else None
        nn_Hs_gt_x.append(nn_H)
        pinx = i + 1
        print(f"nn_T_gt_x[partition {pinx}] type/shape:", 
              type(gt_T), getattr(gt_T, 'shape', None))
        print(f"nn_V_gt_x[partition {pinx}] type/shape:", 
              type(gt_V), getattr(gt_V, 'shape', None))
        print(f"nn_H_gt_x[partition {pinx}] type/shape:", 
              type(gt_H), getattr(gt_H, 'shape', None))
        print('nn_T_gt_x sample values:', nn_T[:5] if nn_T is not None else None)
        print('nn_V_gt_x sample values:', nn_V[:5] if nn_V is not None else None)
        print('nn_H_gt_x sample values:', nn_H[:5] if nn_H is not None else None)

for i in range(len(nn_energy_nn_x_filenames)):
    nn_energy_nn_x_path=data4plot_dir + nn_energy_nn_x_filenames[i]
    print(f"nn_energy_nn_x_path[{i}]:", nn_energy_nn_x_path)
    with np.load(nn_energy_nn_x_path, allow_pickle=False) as data:
        print("keys in .npz:", data.files)
        if 'T' in data.files:
            nn_T = data['T']
        else:
            nn_T = data[data.files[0]]
        nn_Ts_nn_x.append(nn_T)
        if 'V' in data.files:
            nn_V = data['V']
        else:
            nn_V = data[data.files[1]] if len(data.files) > 1 else None
        nn_Vs_nn_x.append(nn_V)
        if 'H' in data.files:
            nn_H = data['H']
        else:
            nn_H = data[data.files[2]] if len(data.files) > 2 else None
        nn_Hs_nn_x.append(nn_H)
        pinx = i + 1
        print(f"nn_T_gt_x[partition {pinx}] type/shape:", 
              type(gt_T), getattr(gt_T, 'shape', None))
        print(f"nn_V_gt_x[partition {pinx}] type/shape:", 
              type(gt_V), getattr(gt_V, 'shape', None))
        print(f"nn_H_gt_x[partition {pinx}] type/shape:", 
              type(gt_H), getattr(gt_H, 'shape', None))
        print('nn_T_gt_x sample values:', nn_T[:5] if nn_T is not None else None)
        print('nn_V_gt_x sample values:', nn_V[:5] if nn_V is not None else None)
        print('nn_H_gt_x sample values:', nn_H[:5] if nn_H is not None else None)

from utils_compare_plot_energy import compare_plot_gt_energy
from utils_compare_plot_energy import compare_plot_nn_energy

def plot_gt_energy_comparison(gt_T_gt_x, gt_V_gt_x, gt_H_gt_x,
                              gt_T_nn_x, gt_V_nn_x, gt_H_nn_x,
                              start_pred_year, pred_time_years=400, 
                              steps_per_year=100,
                              full_test_years=400,
                              model_names=model_names,
                              save_dir=plot_dir, img_save=False):
    start_idx = int(start_pred_year // steps_per_year)
    end_idx = start_idx + int(pred_time_years * steps_per_year)
    t_steps = np.array(range(start_idx, end_idx), dtype=int)
    t_years = start_pred_year + (t_steps // steps_per_year).astype(int)

    print(f"start_idx: {start_idx}, end_idx: {end_idx}")
    print(f"start year: {t_years[0]}, end year: {t_years[-1]}")
    print(f"t_years.shape: {t_years.shape}")

    plt.figure(figsize=(8, 6))

    colors = ["red","magenta", "green", "blue", "orange", "cyan", "purple", "brown"]  
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

    for model_idx in range(len(model_names)):
        if plot_with_T_and_V: 
            plt.plot(t_years, gt_T_nn_x[model_idx], 
                     label=f"Tgt(nn_x_{model_names[model_idx]})", 
                     color="magenta", linestyle='--', 
                     linewidth=0.5)
            plt.plot(t_years, gt_V_nn_x[model_idx], 
                     label=f"Vgt(nn_x_{model_names[model_idx]})", 
                     color="green", linestyle='--', 
                     linewidth=0.5)
            
        plt.plot(t_years, gt_H_nn_x[model_idx], 
                 label=f"Hgt(nn_x_{model_names[model_idx]})", 
                 color=colors[model_idx], 
                 # marker=markers[model_idx],
                 # markersize=3,
                 linestyle=linestyles[model_idx], linewidth=1) # 0.5)

    if plot_with_T_and_V:    
        plt.plot(t_years, gt_T_gt_x[0], label="Tgt(gt_x})", color="blue", linewidth=0.5)
        plt.plot(t_years, gt_V_gt_x[0], label=f"Vgt(gt_x)", color="orange", linewidth=0.5)
    
    plt.plot(t_years, gt_H_gt_x[0], label=f"Hgt(gt_x)", color="black", linewidth=2.0)

    plt.xlabel("Time (year)", fontsize=14)
    plt.ylabel("GT Energy", fontsize=14)
    plt.grid(True)
    plt.legend(loc='best', ncol=1, fontsize=10)
    plt.title(f"GT Energies for P-HNN orbits (evaluation partition 1)", fontsize=16)
    plt.tight_layout()
    if img_save:
        save_path = f"{save_dir}/p-hnn-l4_gt_energy_comparison_part1_{start_pred_year}_{pred_time_years}_years.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_nn_energy_comparison(nn_T_gt_x, nn_V_gt_x, nn_H_gt_x,
                              nn_T_nn_x, nn_V_nn_x, nn_H_nn_x,
                              start_pred_year, pred_time_years=400, 
                              steps_per_year=100,
                              full_test_years=400,
                              model_names=model_names,
                              save_dir=plot_dir, img_save=False):
    start_idx = int(start_pred_year // steps_per_year)
    end_idx = start_idx + int(pred_time_years * steps_per_year)
    t_steps = np.array(range(start_idx, end_idx), dtype=int)
    t_years = start_pred_year + (t_steps // steps_per_year).astype(int)

    print(f"start_idx: {start_idx}, end_idx: {end_idx}")
    print(f"start year: {t_years[0]}, end year: {t_years[-1]}")
    print(f"t_years.shape: {t_years.shape}")

    plt.figure(figsize=(8, 6))

    gt_colors = ["black","red","blue", "green", "magenta"]
    colors = ["red","black","magenta","blue", "orange", "magenta", "green", "blue", "orange", "cyan", "purple", "brown"]  
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    for model_idx in range(len(model_names)):
        plt.plot(t_years, nn_H_gt_x[model_idx], 
                 label=f"Hnn(gt_x) [nn={model_names[model_idx]}]", 
                 color=gt_colors[model_idx], 
                 linestyle='-',
                 linewidth=1.0)
        
        if plot_with_T_and_V:
            plt.plot(t_years, gt_Ts_nn_x[model_idx], 
                     label=f"Tnn(nn_x_{model_names[model_idx]})", 
                     color="magenta", linestyle='-', # '--', 
                     linewidth=1)
            plt.plot(t_years, gt_Vs_nn_x[model_idx], 
                     label=f"Vnn(nn_x_{model_names[model_idx]})", 
                     color="green", 
                     linestyle='-', # '--', 
                     linewidth=0.5)
            
        plt.plot(t_years, nn_H_nn_x[model_idx], 
                 label=f"Hnn(nn_x_{model_names[model_idx]})", 
                 color=colors[model_idx], 
                 # marker=markers[model_idx],
                 # markersize=3,
                 linestyle=':', # linestyles[model_idx], 
                 linewidth=0.5)
        


    plt.xlabel("Time (year)", fontsize=14)
    plt.ylabel("NN Energy", fontsize=14)
    plt.grid(True)
    plt.legend(loc='best', ncol=1, fontsize=10)
    plt.title(f"NN Energies of P-HNNs (evaluation partition 1)", fontsize=16)
    plt.tight_layout()
    if img_save:
        save_path = f"{save_dir}/p-hnn-l4_nn_energy_comparison_part1_{start_pred_year}_{pred_time_years}_years.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()    


plot_gt_energy_comparison(gt_Ts_gt_x, gt_Vs_gt_x, gt_Hs_gt_x,
                          gt_Ts_nn_x, gt_Vs_nn_x, gt_Hs_nn_x,
                          start_pred_year=3200, pred_time_years=400, 
                          steps_per_year=100,
                          full_test_years=400,
                          model_names=model_names,
                          save_dir=plot_dir, img_save=True)

plot_nn_energy_comparison(nn_Ts_gt_x, nn_Vs_gt_x, nn_Hs_gt_x,
                          nn_Ts_nn_x, nn_Vs_nn_x, nn_Hs_nn_x,
                          start_pred_year=3200, pred_time_years=400, 
                          steps_per_year=100,
                          full_test_years=400,
                          model_names=model_names,
                          save_dir=plot_dir, img_save=True)