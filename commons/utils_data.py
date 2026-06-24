
import os
import sys

def import_mltsc_v2_data(model_dir, data_dir_name='dataset', 
                      q_file_name='q.npz', p_file_name='p.npz', 
                      dq_file_name='dq.npz', dp_file_name='dp.npz', 
                      train_ratio=0.8, bsz=1024, scale_stat_postfix=""):
    target_path = os.path.join(os.path.dirname(os.getcwd()), data_dir_name)
    if target_path not in sys.path:
        sys.path.append(target_path)

    from qp_ode_dataset_multiscale_v2 import SolarSystemDataset
    from torch.utils.data import DataLoader

    pos_path = f"{target_path}/{q_file_name}"
    mom_path = f"{target_path}/{p_file_name}"
    dpos_path = f"{target_path}/{dq_file_name}"
    dmom_path = f"{target_path}/{dp_file_name}"

    train_dataset = SolarSystemDataset(pos_path, mom_path, dpos_path, dmom_path, mode='train', 
                                       train_ratio=train_ratio)
    train_stats = train_dataset.get_stats()

    test_dataset = SolarSystemDataset(pos_path, mom_path, dpos_path, dmom_path, mode='test', 
                                      train_ratio=train_ratio, 
                                      stats=train_stats)

    import joblib
    joblib.dump(train_stats, f"{model_dir}/ds_mlt_scalev{scale_stat_postfix}.pkl")

    train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bsz, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader

def import_mltsc_data(model_dir, data_dir_name='data_ode',
                      q_file_name='q.npz', p_file_name='p.npz', 
                      dq_file_name='dq.npz', dp_file_name='dp.npz', 
                      train_ratio=0.8, bsz=1024, scale_stat_postfix=""):
    target_path = os.path.join(os.path.dirname(os.getcwd()), data_dir_name)
    if target_path not in sys.path:
        sys.path.append(target_path)

    from qp_ode_dataset_multiscale import SolarSystemDataset
    from torch.utils.data import DataLoader

    pos_path = f"{target_path}/{q_file_name}"
    mom_path = f"{target_path}/{p_file_name}"
    dpos_path = f"{target_path}/{dq_file_name}"
    dmom_path = f"{target_path}/{dp_file_name}"

    train_dataset = SolarSystemDataset(pos_path, mom_path, dpos_path, dmom_path, mode='train',
                                       train_ratio=train_ratio)
    train_stats = train_dataset.get_stats()

    test_dataset = SolarSystemDataset(pos_path, mom_path, dpos_path, dmom_path, mode='test', 
                                      train_ratio=train_ratio,
                                      stats=train_stats)

    import joblib
    joblib.dump(train_stats, f"{model_dir}/ds_mlt_scalev{scale_stat_postfix}.pkl")

    train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bsz, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader

def import_unisc_data(model_dir, data_dir_name='data_ode', 
                      q_file_name='q.npz', p_file_name='p.npz', 
                      dq_file_name='dq.npz', dp_file_name='dp.npz',
                      train_ratio=0.8, bsz=1024, scale_stat_postfix=""):
    target_path = os.path.join(os.path.dirname(os.getcwd()), data_dir_name)
    if target_path not in sys.path:
        sys.path.append(target_path)

    from qp_ode_dataset_uniscale import SolarSystemDataset
    from torch.utils.data import DataLoader

    pos_path = f"{target_path}/{q_file_name}"
    mom_path = f"{target_path}/{p_file_name}"
    dpos_path = f"{target_path}/{dq_file_name}"
    dmom_path = f"{target_path}/{dp_file_name}"

    train_dataset = SolarSystemDataset(pos_path, mom_path, dpos_path, dmom_path, 
                                      mode='train',
                                      train_ratio=train_ratio)
    train_stats = train_dataset.get_stats()

    test_dataset = SolarSystemDataset(pos_path, mom_path, dpos_path, dmom_path, 
                                      mode='test', 
                                      train_ratio=train_ratio, 
                                      stats=train_stats)

    import joblib
    joblib.dump(train_stats, f"{model_dir}/ds_uni_scalev{scale_stat_postfix}.pkl")

    train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bsz, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader