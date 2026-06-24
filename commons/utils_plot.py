
import numpy as np
import matplotlib.pyplot as plt

def plot_losses(loss_vec, start_epoch, caption='Train Loss', save_path=None):
    plt.figure(figsize=(10, 6))
       
    epochs = range(start_epoch+1, start_epoch+len(loss_vec) + 1)
    
    # Train/Test Loss Plot
    plt.plot(epochs, loss_vec, 'b-o', label=caption, markersize=4)
    
    plt.yscale('log')
    
    plt.title(f'HNN {caption} (log scale)', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    if save_path is not None:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Loss plot saved as: {save_path}")
        except Exception as e:
            print(f"❌ Failed to save plot: {e}")
        
    plt.show()

def get_global_limits(q_path):

    data = np.load(q_path)
    q = data['q']  # (2000001, 10, 3)

    x_min, x_max = q[:, :, 0].min(), q[:, :, 0].max()
    y_min, y_max = q[:, :, 1].min(), q[:, :, 1].max()

    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2

    max_range = max(x_max - x_min, y_max - y_min) / 2

    plt_scale_pars = {'cen_x': center_x, 'cen_y': center_y, 'max_r': max_range}
    
    return plt_scale_pars

def subplot_all_planet_orbits(ax, true_q, model_q, colors, names,
                              plt_scale_pars):
    
    global_center_x = plt_scale_pars['cen_x']
    global_center_y = plt_scale_pars['cen_y']
    global_max_range = plt_scale_pars['max_r']

    ax.set_facecolor("#000000")
    for i in range(10):
        idx_x, idx_y = i*3, i*3+1
        ax.plot(true_q[:, idx_x], true_q[:, idx_y], color=colors[i], 
                linestyle='-', alpha=1.0, linewidth=0.25, label=f"{names[i]} (True)")  # alpha=0.3,
        ax.plot(model_q[:, idx_x], model_q[:, idx_y], color=colors[i], 
                linestyle='--', alpha=1.0, linewidth=0.5, label=f"{names[i]} (Model)") # alpha=0.8,

        ax.scatter(true_q[0, idx_x], true_q[0, idx_y], 
                    color=colors[i], 
                    s=10,
                    edgecolors='white',
                    marker='o',
                    zorder=10, 
                    label=f'Start {names[i]}')

    padding = 1.05
    ax.set_xlim(global_center_x - global_max_range * padding, 
                global_center_x + global_max_range * padding)
    ax.set_ylim(global_center_y - global_max_range * padding, 
                global_center_y + global_max_range * padding)
    
    ax.set_aspect('equal')

    ax.set_title('Solar System Orbit', color='black', fontsize=8, pad=5)#, fontweight='bold')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
              fontsize=3.5, facecolor='#1e1e1e', labelcolor='white',
              edgecolor='white', framealpha=0.8, handlelength=3)

    ax.tick_params(colors='black', labelsize=4)

    ax.set_xlabel('X (AU)', color='black', fontsize=4)
    ax.set_ylabel('Y (AU)', color='black', fontsize=4)

    ax.grid(False)
    #ax.grid(color='gray', linestyle='--', alpha=0.1)

def subplot_Sun_to_Mars_orbits(ax, true_q, model_q, colors, names):
    ax.set_facecolor("#000000")
    for i in range(5):
        idx_x, idx_y = i*3, i*3+1
        ax.plot(true_q[:, idx_x], true_q[:, idx_y], color=colors[i], 
                linestyle='-', alpha=1.0, linewidth=0.25, label=f"{names[i]} (True)")  # alpha=0.3,
        ax.plot(model_q[:, idx_x], model_q[:, idx_y], color=colors[i], 
                linestyle='--', alpha=1.0, linewidth=0.5, label=f"{names[i]} (Model)") # alpha=0.8,

        ax.scatter(true_q[0, idx_x], true_q[0, idx_y], 
                   color=colors[i], 
                   s=10,
                   edgecolors='white',
                   marker='o',
                   zorder=10, 
                   label=f'Start {names[i]}')
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
    
        ax.set_aspect('equal')

        ax.set_title('Sun-to-Mars Orbit', color='black', fontsize=8, pad=5)#, fontweight='bold')
    
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                  fontsize=3.5, facecolor='#1e1e1e', labelcolor='white',
                  edgecolor='white', framealpha=0.8, handlelength=3)
    
        ax.tick_params(colors='black', labelsize=4)
    
        ax.set_xlabel('X (AU)', color='black', fontsize=4)
        ax.set_ylabel('Y (AU)', color='black', fontsize=4)

        ax.grid(False)
        #ax.grid(color='gray', linestyle='--', alpha=0.1)

def plot_orbits(true_q, model_q, 
                plt_scale_pars,
                save_path, img_save=False):
    
    global_center_x = plt_scale_pars['cen_x']
    global_center_y = plt_scale_pars['cen_y']
    global_max_range = plt_scale_pars['max_r']

    if model_q.ndim == 3: model_q = model_q.squeeze(1)
    if true_q.ndim == 3: true_q = true_q.squeeze(1)

    nick_names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
    colors = ["#FFD700", "#1E90FF", "#FF4500", "#32CD32", "#FF8C00", "#8A2BE2", "#00CED1", "#FF00FF", "#7FFF00", "#FFFFFF"]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))

    subplot_all_planet_orbits(axes[0], true_q, model_q, colors, nick_names,
                              plt_scale_pars=plt_scale_pars)
    subplot_Sun_to_Mars_orbits(axes[1], true_q, model_q, colors, nick_names)

    plt.tight_layout()
    if img_save:
        plt.savefig(save_path, facecolor='#ffffff', dpi=300, format='webp')
    plt.show()

def plot_metrics(true_q, true_p, model_q, model_p, 
                 model, device, q_max, p_max,
                 start_time, dt, 
                 qp_cat=True, 
                 save_path=None, 
                 img_save=False):

    from utils_energy import calculate_gt_hamiltonian
    from utils_energy import calculate_nn_hamiltonian

    if model_q.ndim == 3: model_q = model_q.squeeze(1)
    if model_p.ndim == 3: model_p = model_p.squeeze(1)
    if true_q.ndim == 3: true_q = true_q.squeeze(1)
    if true_p.ndim == 3: true_p = true_p.squeeze(1)

    steps = len(true_q)
    times = np.arange(start_time, start_time + steps * dt, dt)

    nick_names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
    colors = ["#FFD700", "#1E90FF", "#FF4500", "#32CD32", "#FF8C00", "#8A2BE2", "#00CED1", "#FF00FF", "#7FFF00", "#000000"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3)) #, facecolor='#1e1e1e')
    
    # --- 1. Hamiltonian (Energy) ---
    ax_h = axes[0]
    Hnn_gt_x = calculate_nn_hamiltonian(model, device, true_q, true_p, q_max, p_max, qp_cat)
    Hnn_nn_x = calculate_nn_hamiltonian(model, device, model_q, model_p, q_max, p_max, qp_cat)
    Hgt_gt_x = calculate_gt_hamiltonian(true_q, true_p)
    Hgt_nn_x = calculate_gt_hamiltonian(model_q, model_p)

    ax_h.plot(times, Hgt_gt_x, color='black', label='Hgt(gt_x)', alpha=1.0) # alpha=0.6
    ax_h.plot(times, Hgt_nn_x, color='blue', label='Hgt(nn_x)', alpha=1.0) #, linestyle='--') # alpha=0.8
    ax_h.plot(times, Hnn_gt_x, color='green', label='Hnn(gt_x)', alpha=1.0) # alpha=0.6
    ax_h.plot(times, Hnn_nn_x, color='red', label='Hnn(nn_x)', alpha=1.0) #, linestyle='--') # alpha=0.8

    ax_h.set_title('Hamiltonian', color='black', fontsize=10) # 이전값 fontsize=15
    ax_h.set_xlabel('Time (Year)', color='black')
    # ax_h.legend(loc='upper left', fontsize=18, ncol=2) 
    ax_h.legend(fontsize=6) #, ncol=2)
    ax_h.grid(alpha=1.0) # 0.1
    ax_h.tick_params(colors='black')

    # --- 2. Position MSE (Q) ---
    ax_mq = axes[1]
    for i in range(10):
        idx = slice(i*3, (i+1)*3)
        mse_qi = np.mean((true_q[:, idx] - model_q[:, idx])**2, axis=1)
        ax_mq.plot(times, mse_qi, color=colors[i], alpha=1.0, label=f"{nick_names[i]}") # , alpha=0.7)
    ax_mq.set_yscale('log')
    ax_mq.set_title('Position MSE (Log Scale)', fontsize=10, color='black')
    ax_mq.set_xlabel('Time (Year)')
    ax_mq.grid(True, which="both", alpha=1.0)
    ax_mq.legend(loc='lower right', fontsize=6, ncol=2)

    # --- 3. Momentum MSE (P) ---
    ax_mp = axes[2]
    for i in range(10):
        idx = slice(i*3, (i+1)*3)
        mse_pi = np.mean((true_p[:, idx] - model_p[:, idx])**2, axis=1)
        ax_mp.plot(times, mse_pi, color=colors[i], alpha=1.0, label=f"{nick_names[i]}") # , alpha=0.7)
    ax_mp.set_yscale('log')
    ax_mp.set_title('Momentum MSE (Log Scale)', fontsize=10, color='black')
    ax_mp.set_xlabel('Time (Year)')
    ax_mp.grid(True, which="both", alpha=1.0)
    ax_mp.legend(loc='lower right', fontsize=6, ncol=2)

    plt.tight_layout()
    if img_save and save_path is not None:
        plt.savefig(save_path, facecolor='#ffffff', dpi=300)
    plt.show()

def check_eval_on_fit_set(model, device, 
                         q_max, p_max, dq_max, dp_max, dt, 
                         fit_steps, gt_q_full, gt_p_full,
                         prefix, weight_filename, fit_fig_dir,
                         plt_scale_pars,
                         img_save=False):

    from utils_orbit import scipy_predict_segment

    fit = {}

    begin_idx = 0
    for i, end_idx in enumerate(fit_steps):
        gt_q_i, gt_p_i, nn_q_i, nn_p_i = scipy_predict_segment(model, device, 
                                                              gt_q_full, gt_p_full, 
                                                              begin_idx, end_idx, dt,
                                                              q_max, p_max, dq_max, dp_max)
    
        fit[f"gt_q[{i}]"]=gt_q_i
        fit[f"gt_p[{i}]"]=gt_p_i
        fit[f"nn_q[{i}]"]=nn_q_i
        fit[f"nn_p[{i}]"]=nn_p_i

        print('qt_q_i shape:', gt_q_i.shape)
        print('nn_q_i shape:', nn_q_i.shape)
        
        fit_filename = f"{prefix}fit_[{begin_idx} : {end_idx}]_{weight_filename}" + ".webp"
        print(fit_filename)
        plot_orbits(fit[f"gt_q[{i}]"], fit[f"nn_q[{i}]"],
                    plt_scale_pars=plt_scale_pars,
                    save_path=f"{fit_fig_dir}/" + fit_filename, 
                    img_save=img_save)
    
        fit_stats_filename = f"{prefix}fit_stats_[{begin_idx} : {end_idx}]_{weight_filename}" + ".webp"
        print(fit_stats_filename)
        plot_metrics(fit[f"gt_q[{i}]"], fit[f"gt_p[{i}]"], 
                     fit[f"nn_q[{i}]"], fit[f"nn_p[{i}]"], 
                     model, device, q_max, p_max, 
                     start_time=begin_idx*dt, dt=dt,
                     save_path=f"{fit_fig_dir}/" +fit_stats_filename,
                     img_save=img_save)
    
        begin_idx=end_idx
    

def eval_on_test_set(model, device, 
                     q_max, p_max, dq_max, dp_max, dt, 
                     eval_steps, gt_q, gt_p, start_idx, 
                     prefix, weight_filename, eval_fig_dir, 
                     plt_scale_pars,
                     qp_cat=True,
                     img_save=False):

    from utils_orbit import scipy_predict_qp_model_orbits

    q_init_val = gt_q[0:1]
    p_init_val = gt_p[0:1]
    
    eval = {}
    launch_idx = 0
    begin_idx = 0
    
    for i, end_idx in enumerate(eval_steps):
        print(f"i: {i}, interval=[{launch_idx} : {end_idx}]")
        stp = end_idx - begin_idx
        nn_q_i, nn_p_i = scipy_predict_qp_model_orbits(model, device, 
                                                       q_init_val, p_init_val, stp, dt,
                                                       q_max, p_max, dq_max, dp_max) 

        gt_qkey = f"gt_q[{i}]"
        gt_pkey = f"gt_p[{i}]"
        nn_qkey = f"nn_q[{i}]"
        nn_pkey = f"nn_p[{i}]"

        if eval == {}:
           eval[gt_qkey]=gt_q[begin_idx:end_idx]
           eval[gt_pkey]=gt_p[begin_idx:end_idx]
           eval[nn_qkey]=nn_q_i
           eval[nn_pkey]=nn_p_i
        else:
           gt_q_prev = eval[gt_qkey_prev]
           gt_p_prev = eval[gt_pkey_prev]
           nn_q_prev = eval[nn_qkey_prev]
           nn_p_prev = eval[nn_pkey_prev]
           eval[gt_qkey]=np.concatenate([gt_q_prev, gt_q[begin_idx:end_idx]], axis=0)
           eval[gt_pkey]=np.concatenate([gt_p_prev, gt_p[begin_idx:end_idx]], axis=0)
           eval[nn_qkey]=np.concatenate([nn_q_prev, nn_q_i], axis=0)
           eval[nn_pkey]=np.concatenate([nn_p_prev, nn_p_i], axis=0)

        orbit_filename = f"{prefix}eval_[{launch_idx} : {end_idx}]_{weight_filename}" + ".webp"
        print(orbit_filename)
        plot_orbits(eval[f"gt_q[{i}]"], eval[f"nn_q[{i}]"], 
                    save_path=f"{eval_fig_dir}/" + orbit_filename,
                    plt_scale_pars=plt_scale_pars,
                    img_save=img_save)

        metric_filename = f"{prefix}metric_[{launch_idx} : {end_idx}]_{weight_filename}" + ".webp"
        print(metric_filename)
        plot_metrics(eval[f"gt_q[{i}]"], eval[f"gt_p[{i}]"], 
                     eval[f"nn_q[{i}]"], eval[f"nn_p[{i}]"], 
                     model, device, q_max, p_max, 
                     start_time=start_idx*dt, dt=dt,
                     qp_cat=qp_cat,
                     save_path=f"{eval_fig_dir}/" + metric_filename,
                     img_save=img_save)     
                                              
        begin_idx=end_idx
        gt_qkey_prev = gt_qkey
        gt_pkey_prev = gt_pkey
        nn_qkey_prev = nn_qkey
        nn_pkey_prev = nn_pkey
        q_init_val = nn_q_i[(stp-1):stp]
        p_init_val = nn_p_i[(stp-1):stp]
