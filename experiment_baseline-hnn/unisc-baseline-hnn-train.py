import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt
import pickle

common_dir_name = 'commons' 
common_dir_path = os.path.join(os.path.dirname(os.getcwd()), common_dir_name)
if common_dir_path not in sys.path:
    sys.path.append(common_dir_path)

from utils_train import seed_everything
from utils_train import init_weights
from utils_data import import_unisc_data
from utils_chkpt import save_checkpoint
from utils_train import train_one_epoch, test_one_epoch
from nn_model_baseline import HNN

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LAYERS = 3
HIDDEN_DIMS = 240

BATCH_SZ = 8192
LR_INIT = 1e-2
MIN_LR = 1e-6
PATIENCE = 3

PREFIX=""
POSTFIX=f"_lr_{LR_INIT:.0e}_pat_{PATIENCE}"

PREV_EPOCS = -1
EXTRA_EPOCS = 100

MODEL_NAME=f"unisc-baseline-hnn({LAYERS})({HIDDEN_DIMS})"
MODEL_DIR="./checkpoints/uni-scale"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

PLOT_ROOT_DIR="./figures"
TYPE_DIR=f"{PLOT_ROOT_DIR}/loss"
FIG_DIR=f"{TYPE_DIR}/uni-scale"

if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR, exist_ok=True)

start_epoch = 0
best_loss = float('inf')

train_loader, test_loader = None, None
model, optimizer, scheduler = None, None, None

history = {
    'train_losses': [],
    'test_losses': [],
    'learn_rates': []
}

def prepare_all(hidden_dim=240, layers=3):
    global train_loader, test_loader
    global model, optimizer, scheduler
    global start_epoch, best_loss
    global OUTPUT_PREFIX, OUTPUT_POSTFIX, new_sched, NEW_SCHED

    seed_everything(0)

    train_loader, test_loader=import_unisc_data(model_dir=MODEL_DIR, 
                                                data_dir_name='data_ode',
                                                q_file_name='q.npz', 
                                                p_file_name='p.npz',
                                                dq_file_name='dq.npz', 
                                                dp_file_name='dp.npz',
                                                train_ratio=0.8, # 8:2 ratio
                                                bsz=BATCH_SZ)

    weight_filename = f"{MODEL_NAME}_ep_{PREV_EPOCS}_of_{PREV_EPOCS}{POSTFIX}"
    checkpoint_path = f"{MODEL_DIR}/" + weight_filename + '.pth'
    print('checkpoint path:', checkpoint_path)

    model = HNN(hidden_dim=hidden_dim, num_layers=layers)
    model = model.to(device)
    model.double() # double precision
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)
    dt = 0.01 # 1 year = 100 steps 

    print(model)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('loss', float('inf'))
        print(f"Resuming from Epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting from scratch.")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=PATIENCE, 
        threshold=1e-4,
        min_lr=MIN_LR,
    )

    # set some compatibility aliases used later in the script
    OUTPUT_PREFIX = PREFIX
    OUTPUT_POSTFIX = POSTFIX
    # scheduling mode: 'pat' == ReduceLROnPlateau, 'ca' == CosineAnnealing (example)
    NEW_SCHED = 'pat'
    new_sched = NEW_SCHED

def train_with_val(start_epoch=start_epoch, num_extra_epochs=100):
    global best_loss
    global history
    
    total_epochs = start_epoch + num_extra_epochs
     
    for epoch in range(start_epoch, total_epochs):
        print(f'epoc: {epoch+1}/{total_epochs}')

        # --- [TRAIN PHASE] ---
        avg_train_loss=train_one_epoch(model, optimizer, train_loader, device, 
                                       grad_max_norm=0.5, grad_clipping=False)
        history['train_losses'].append(avg_train_loss)

        # --- [TEST/VAL PHASE] ---
        avg_test_loss=test_one_epoch(model, test_loader, device)
        history['test_losses'].append(avg_test_loss)
        
        print(f"\n" + "="*50)
        print(f"Epoch {epoch+1} Summary:")
        print(f"Average Train Loss: {avg_train_loss:.7e}")
        print(f"Average Test Loss:  {avg_test_loss:.7e}")
        print("="*50 + "\n")

        # --- [BEST MODEL SAVING] ---
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            save_checkpoint(
                model, optimizer, total_epochs, epoch, avg_test_loss, 
                prefix=OUTPUT_PREFIX, 
                postfix=OUTPUT_POSTFIX, 
                path=MODEL_DIR,
                name=MODEL_NAME
            )
            print(f"🌟 Best Test Loss Updated! Model Saved.\n")

        if epoch == total_epochs - 1:
            save_checkpoint(
                model, optimizer, total_epochs, epoch, avg_test_loss, 
                prefix=OUTPUT_PREFIX, 
                postfix=OUTPUT_POSTFIX, 
                path=MODEL_DIR,
                name=MODEL_NAME
            )
            print(f"Epoch {epoch+1} Final Model Saved.")

        if new_sched == 'pat':
            scheduler.step(avg_test_loss)
            current_lr = optimizer.param_groups[0]['lr']
        elif new_sched == 'ca':    
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        print(f"DEBUG | Best: {scheduler.best:.4e} | Bad: {scheduler.num_bad_epochs}")
            
        history['learn_rates'].append(current_lr)

        print(f"Epoch {epoch+1} finished | Test Loss: {avg_test_loss:.7e} | Current LR: {current_lr}")

def plot_loss_history(history, save_path="./figures/hnn_loss_plot.png"):
    plt.figure(figsize=(10, 6))
    
    epochs = range(start_epoch+1, start_epoch+len(history['train_losses']) + 1)
    
    # Train/Test Loss Plot
    plt.plot(epochs, history['train_losses'], 'b-o', label='Train Loss', markersize=4)
    plt.plot(epochs, history['test_losses'], 'r-s', label='Test Loss', markersize=4)
    
    plt.yscale('log')
    
    plt.title('HNN Training & Test Loss (log scale)', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Loss plot saved as: {save_path}")
    except Exception as e:
        print(f"❌ Failed to save plot: {e}")
        
    plt.show()


def process_post_train():
    start_idx = start_epoch + 1
    end_idx = start_idx + len(history['train_losses']) - 1
    filename = f"loss_{MODEL_NAME}_" \
               f"ep_{start_idx}_to_{end_idx}" \
               + OUTPUT_POSTFIX + ".webp"

    save_dir = FIG_DIR
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plot_loss_history(history, save_path=save_path)

    start_idx = start_epoch + 1
    end_idx = start_idx + len(history['train_losses']) - 1
    filename = f"stats_{MODEL_NAME}_" \
           f"ep_{start_idx}_to_{end_idx}_bs_{BATCH_SZ}_lr_{LR_INIT:.0e}" \
           f"_pat_{PATIENCE}.pkl"

    save_dir = MODEL_DIR
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    with open(save_path, 'wb') as f:
	    pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=='__main__':
    prepare_all(hidden_dim=HIDDEN_DIMS, layers=LAYERS)
    train_with_val(start_epoch=start_epoch, num_extra_epochs=EXTRA_EPOCS)
    process_post_train()