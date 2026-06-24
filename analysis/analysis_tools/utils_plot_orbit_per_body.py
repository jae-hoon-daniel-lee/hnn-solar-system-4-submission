
import matplotlib.pyplot as plt
import numpy as np

planets = ["Sun",
            "Mercury",
            "Venus",
            "Earth",
            "Mars",
            "Jupiter",
            "Saturn",
            "Uranus",
            "Neptune",
            "Pluto"
            ]

def compare_plot2d_orbits_per_planet(q_pred, q_true, 
                                     planets=planets, pred_years=50,
                                     sample_steps=1,
                                     num_partitions=1, 
                                     partition_idx=0,
                                     model_name="", 
                                     save_dir=".", 
                                     img_save=False):
    step = sample_steps

    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    for i in range(10):
        axes[i].plot(q_true[::step, i, 0], q_true[::step, i, 1], "-", color="white", label="True")
        axes[i].plot(q_pred[::step, i, 0], q_pred[::step, i, 1], "--", color="red", label="Predicted", alpha=0.5)
        axes[i].scatter(q_true[-1, i, 0], q_true[-1, i, 1], color="white", s=40)
        axes[i].scatter(q_pred[-1, i, 0], q_pred[-1, i, 1], color="red", s=40)
        axes[i].set_title(planets[i], fontsize=28, pad=10)
        axes[i].axis('equal') 
        axes[i].set_facecolor('black')
        axes[i].legend(loc="best", fontsize='9', frameon=True, framealpha=0.8, shadow=True)
    if num_partitions == 1:
        title = f"Orbits for {pred_years} years"
    elif num_partitions > 1:
        p = partition_idx + 1
        title = f"Orbits for {pred_years} years [partition: {p}]"
    else:
        title = ""
        
    fig.suptitle(title, fontweight="bold", fontsize=32, y=0.95) 
    plt.tight_layout(rect=[0, 0, 1, 0.91]) 
    if img_save:
        if num_partitions == 1:
            save_path = f"{save_dir}/{model_name}-2d_orbits_{pred_years}_years"\
                        ".webp"
        elif num_partitions > 1:
            p = partition_idx + 1
            save_path = f"{save_dir}/{model_name}-2d_orbits_{pred_years}_years_part{p}"\
                        ".webp"
        else:
            plt.show()
            return
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compare_plot3d_orbits_per_planet(q_pred, q_true, 
                                     planets=planets, 
                                     pred_years=50,
                                     sample_steps=1,
                                     num_partitions=1, 
                                     partition_idx=0,
                                     model_name="", 
                                     save_dir=".", 
                                     img_save=False):
    import numpy as np
    import matplotlib.pyplot as plt

    step = sample_steps
    padding = 1.05

    fig, ax = plt.subplots(5, 2, figsize=(18, 40), subplot_kw={'projection': '3d'})
    fig.set_facecolor('#c3c3c3')
    ax = ax.flatten()

    for i in range(len(planets)):
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_zticklabels([])

        # plot orbits
        ax[i].plot(q_pred[::step, i, 0], q_pred[::step, i, 1], q_pred[::step, i, 2],
                   "-", color="red", label="Predicted")
        ax[i].plot(q_true[::step, i, 0], q_true[::step, i, 1], q_true[::step, i, 2],
                   "--", color="k", label="True", alpha=0.5)
        ax[i].plot(q_pred[-1, i, 0], q_pred[-1, i, 1], q_pred[-1, i, 2], "o", color="red")
        ax[i].plot(q_true[-1, i, 0], q_true[-1, i, 1], q_true[-1, i, 2], "o", color="k")

        try:
            coords_pred = q_pred[:, i, :]   # shape (T, 3)
            coords_true = q_true[:, i, :]
            coords = np.concatenate([coords_pred, coords_true], axis=0)  # (2T, 3)
        except Exception:
            coords = np.vstack([np.reshape(q_pred[:, i, :], (-1, 3)),
                                np.reshape(q_true[:, i, :], (-1, 3))])

        min_coords = np.nanmin(coords, axis=0)
        max_coords = np.nanmax(coords, axis=0)
        spans = max_coords - min_coords
        max_span = np.nanmax(spans)
        if np.isnan(max_span) or max_span == 0:
            max_span = 1.0

        half = 0.5 * max_span * padding
        center = 0.5 * (max_coords + min_coords)

        x_min, x_max = center[0] - half, center[0] + half
        y_min, y_max = center[1] - half, center[1] + half
        z_min, z_max = center[2] - half, center[2] + half

        ax[i].set_xlim(x_min, x_max)
        ax[i].set_ylim(y_min, y_max)
        ax[i].set_zlim(z_min, z_max)

        try:
            ax[i].set_aspect('equal')
        except Exception:
            try:
                ax[i].set_box_aspect([1, 1, 1])
            except Exception:
                pass

        ax[i].set_title(f"{planets[i]}", fontsize=28, pad=10)
        ax[i].set_xlabel("X", fontsize=16)
        ax[i].set_ylabel("Y", fontsize=16)
        ax[i].set_zlabel("Z", fontsize=16)
        ax[i].legend(loc="best", fontsize='20', frameon=True, framealpha=0.8, shadow=True)

        ax[i].view_init(elev=45, azim=45)

    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    
    if num_partitions == 1:
        title = f"Orbits for {pred_years} years"
    elif num_partitions > 1:
        p = partition_idx + 1
        title = f"Orbits for {pred_years} years [partition: {p}]"
    else:
        title = ""
        
    fig.suptitle(title, fontweight="bold", fontsize=28, y=0.91)
    if img_save:
        if num_partitions == 1:
            save_path = f"{save_dir}/{model_name}-3d_orbits_{pred_years}_years.webp"
        elif num_partitions > 1:
            p = partition_idx + 1
            save_path = f"{save_dir}/{model_name}-3d_orbits_{pred_years}_years_part{p}.webp"
        else:
            plt.show()
            return
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
