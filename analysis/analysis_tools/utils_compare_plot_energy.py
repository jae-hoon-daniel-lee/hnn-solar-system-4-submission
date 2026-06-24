
import matplotlib.pyplot as plt
import numpy as np

def compare_plot_gt_energy(gt_T_gt_x, gt_V_gt_x, gt_H_gt_x, 
                           gt_T_nn_x, gt_V_nn_x, gt_H_nn_x,
                           start_pred_year, 
                           pred_time_years=50, 
                           steps_per_year=2000000/1000,
                           full_test_years=4000,
                           num_partitions=1,
                           save_dir=".", 
                           img_save=False):
                           
    start_idx = int(start_pred_year // steps_per_year)
    end_idx = start_idx + int(pred_time_years * steps_per_year)
    t_steps = np.array(range(start_idx, end_idx), dtype=int)
    t_years = start_pred_year + (t_steps // steps_per_year).astype(int)

    print(f"start_idx: {start_idx}, end_idx: {end_idx}")
    print(f"start year: {t_years[0]}, end year: {t_years[-1]}")
    print(f"t_years.shape: {t_years.shape}")

    for partition_idx in range(num_partitions):
        t_years = t_years + partition_idx * (full_test_years // num_partitions)
        plt.figure(figsize=(8, 6))
        plt.plot(t_years, gt_T_gt_x[partition_idx], label="Tgt(gt_x)", color="blue", linewidth=0.5)
        plt.plot(t_years, gt_V_gt_x[partition_idx], label="Vgt(gt_x)", color="orange", linewidth=0.5)
        plt.plot(t_years, gt_H_gt_x[partition_idx], label="Hgt(gt_x)", color="black", linewidth=0.5)
        plt.plot(t_years, gt_T_nn_x[partition_idx], label="Tgt(nn_x)", color="magenta", linestyle='--', linewidth=0.5)
        plt.plot(t_years, gt_V_nn_x[partition_idx], label="Vgt(nn_x)", color="green", linestyle='--', linewidth=0.5)
        plt.plot(t_years, gt_H_nn_x[partition_idx], label="Hgt(nn_x)", color="red", linestyle='--', linewidth=0.5)
        plt.xlabel("Time (year)")
        plt.ylabel("GT Energy")
        plt.title("GT Energy Comparison")
        plt.legend(loc='best', ncol=6, fontsize=8)
        plt.grid(True)

        if num_partitions == 1:
            title = f"GT Energy for {pred_time_years} years)"
        elif num_partitions > 1:
            p = partition_idx + 1
            title = f"GT Energy for {pred_time_years} years [partition: {p}]"
        else:
            title = ""

        plt.title(title, fontsize=14) 
        plt.tight_layout() 

        if img_save:
            if num_partitions == 1:
                save_path = f"{save_dir}/gt_energy_comparison_{start_pred_year}_{pred_time_years}_years"\
                            ".webp"
            elif num_partitions > 1:
                p = partition_idx + 1
                save_path = f"{save_dir}/gt_energy_comparison_{start_pred_year}_{pred_time_years}_years_part{p}"\
                            ".webp"
            else:
                plt.show()
                return
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
            
                  
                  
def compare_plot_nn_energy(nn_T_gt_x, nn_V_gt_x, nn_H_gt_x, 
                           nn_T_nn_x, nn_V_nn_x, nn_H_nn_x,
                           start_pred_year, 
                           pred_time_years=50, 
                           steps_per_year=2000000/1000,
                           full_test_years=4000,
                           num_partitions=1,
                           save_dir=".", 
                           img_save=False):
                           
    start_idx = int(start_pred_year // steps_per_year)
    end_idx = start_idx + int(pred_time_years * steps_per_year)
    t_steps = np.array(range(start_idx, end_idx), dtype=int)
    t_years = start_pred_year + (t_steps // steps_per_year).astype(int)

    print(f"start_idx: {start_idx}, end_idx: {end_idx}")
    print(f"start year: {t_years[0]}, end year: {t_years[-1]}")
    print(f"t_years.shape: {t_years.shape}")

    for partition_idx in range(num_partitions):
        t_years = t_years + partition_idx * int(full_test_years // num_partitions)
        plt.figure(figsize=(8, 6))
        plt.plot(t_years, nn_T_gt_x[partition_idx], label="Tnn(gt_x)", color="blue", linewidth=0.5)
        plt.plot(t_years, nn_V_gt_x[partition_idx], label="Vnn(gt_x)", color="orange", linewidth=0.5)
        plt.plot(t_years, nn_H_gt_x[partition_idx], label="Hnn(gt_x)", color="black", linewidth=0.5)
        plt.plot(t_years, nn_T_nn_x[partition_idx], label="Tnn(nn_x)", color="magenta", linestyle='--', linewidth=0.5)
        plt.plot(t_years, nn_V_nn_x[partition_idx], label="Vnn(nn_x)", color="green", linestyle='--', linewidth=0.5)
        plt.plot(t_years, nn_H_nn_x[partition_idx], label="Hnn(nn_x)", color="red", linestyle='--', linewidth=0.5)
        plt.xlabel("Time (year)")
        plt.ylabel("NN Energy")
        plt.title("NN Energy Comparison")
        plt.legend(loc='best', ncol=6, fontsize=8)
        plt.grid(True)

        if num_partitions == 1:
            title = f"NN Energy for {pred_time_years} years)"
        elif num_partitions > 1:
            p = partition_idx + 1
            title = f"NN Energy for {pred_time_years} years [partition: {p}]"
        else:
            title = ""

        plt.title(title, fontsize=14) 
        plt.tight_layout() 
 
        if img_save:
            if num_partitions == 1:
                save_path = f"{save_dir}/nn_energy_comparison_{start_pred_year}_{pred_time_years}_years"\
                            ".webp"
            elif num_partitions > 1:
                p = partition_idx + 1
                save_path = f"{save_dir}/nn_energy_comparison_{start_pred_year}_{pred_time_years}_years_part{p}"\
                            ".webp"
            else:
                plt.show()
                return
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
