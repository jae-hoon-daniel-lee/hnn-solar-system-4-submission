
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

checkpoint_dir_path = '../experiment_block-plausible-hnn_lite/checkpoints/multi-scale-v2'

model_prefix='opt-att-'
model_names=['blk-p-hnn(4)(7)']
at_epocs=[36]
facs = [7]

from nn_model_block_plausible import BlockPlausibleHNN
from utils_chkpt import load_checkpoint

attention_vector = []

for model_name, epoch, fac in zip(model_names, at_epocs, facs):
    print(f"Processing model: {model_name}_ep{epoch}")
    
    # Load the model checkpoint
    model = BlockPlausibleHNN(m=fac, n_layers=4).to(device)
    model.double()
    checkpoint_path = f"{checkpoint_dir_path}/{model_prefix}{model_name}-bs200" \
                      f"_ep_{epoch}_of_60_lr_1e-04_pat_3.pth"
    _, _ = load_checkpoint(model, optimizer=None, file_path=checkpoint_path)

    attention_vector.append(model.w_v.cpu().detach().numpy())
    print(f"V-net attention vector: {attention_vector[-1]}")

bodies = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']

import itertools
pairs = list(itertools.combinations(range(len(bodies)), 2))

att_vec_len = len(attention_vector[0])
pair_tuple_len = len(pairs)
print(f'len(attention_vector): {att_vec_len}, len(pairs): {pair_tuple_len}')

if len(attention_vector) != len(pairs):
    print("Warning: The number of attention vectors does not match the number of planet pairs.")
    exit


attention_matrix = np.zeros((len(bodies), len(bodies)))

for idx, pair in enumerate(pairs):
    i, j = pair
    print(f"Attention weight for {bodies[i]}-{bodies[j]}: {[attn[i] for attn in attention_vector]}")
    attention_matrix[i, j] = attention_vector[0][idx]
    attention_matrix[j, i] = attention_vector[0][idx]

vmin = np.min(attention_vector[0], axis=-1)
vmax = np.max(attention_vector[0], axis=-1)

plt.figure(figsize=(6, 6))

ax = plt.gca()

# Create a masked array so diagonal elements can be drawn separately.
mask = np.eye(len(bodies), dtype=bool)
off_diag = np.ma.array(attention_matrix, mask=mask)

# Plot off-diagonal with colormap
im = ax.imshow(off_diag, cmap='viridis', 
               vmin=vmin, vmax=vmax, 
               origin='lower', aspect='equal')

cb = plt.colorbar(im, ax=ax, label='Attention Weight')

# Overlay diagonal as black squares. 
diag_coords = np.arange(len(bodies))
ax.scatter(diag_coords, diag_coords, marker='s', s=200, c='k')

# Ticks and labels
ax.set_xticks(np.arange(len(bodies)))
ax.set_xticklabels(bodies, rotation=45)
ax.set_yticks(np.arange(len(bodies)))
ax.set_yticklabels(bodies)

ax.set_title('V-net Attention for Planet Pairs', fontsize=18)
plt.tight_layout()
plt.savefig("./plots/blk-p-hnn-V-net_attention_matrix.webp")
plt.show()



G_cnst = 4*(np.pi)**2  #(AU^3/(M_sun*yr^2)))

mass=np.array([
    1988410E24,     # Sun
    3.302E23,       # Mercury
    48.685E23,      # Venus
    5.97219E24,     # Earth
    6.4171E23,      # Mars
    18.9819E26,     # Jupiter
    5.6834E26,      # Saturn
    86.813E24,      # Uranus
    102.409E24,     # Neptune
    1.303E22        # Pluto
    ])/1988410E24  # 1.98841E30 kg = 1 M_sun

true_pair_coeff_matrix = np.zeros((len(bodies), len(bodies)))
for i in range(len(bodies)-1):
    j_range=np.linspace(i+1, len(bodies)-1).astype(int)
    for j in j_range:
        print(f'i: {i}, j: {j}')
        true_pair_coeff_matrix[i,j] = np.log(G_cnst * (mass[i] * mass[j]))
        true_pair_coeff_matrix[j,i] = true_pair_coeff_matrix[i,j]

vmin = np.min(true_pair_coeff_matrix)
vmax = np.max(true_pair_coeff_matrix)

plt.figure(figsize=(6, 6))

ax = plt.gca()

# Create a masked array so diagonal elements can be drawn separately.
mask = np.eye(len(bodies), dtype=bool)
off_diag = np.ma.array(true_pair_coeff_matrix, mask=mask)

# Plot off-diagonal with colormap
im = ax.imshow(off_diag, cmap='viridis', 
               vmin=vmin, vmax=vmax, 
               origin='lower', aspect='equal')

cb = plt.colorbar(im, ax=ax, label='log(True Coefficient)')

# Overlay diagonal as black squares.
diag_coords = np.arange(len(bodies))
ax.scatter(diag_coords, diag_coords, marker='s', s=200, c='k')

# Ticks and labels
ax.set_xticks(np.arange(len(bodies)))
ax.set_xticklabels(bodies, rotation=45)
ax.set_yticks(np.arange(len(bodies)))
ax.set_yticklabels(bodies)

ax.set_title('True Pairwse Coeffients', fontsize=18)
plt.tight_layout()
plt.savefig("./plots/real_pair_gravity_V_coeff_matrix.webp")
plt.show()