
import numpy as np
import torch

def get_orbit(model, device, q_init, p_init, steps, dt, 
              q_max, p_max, dq_max, dp_max):
    
    import scipy

    tend = dt * steps
    t_eval = np.linspace(0, tend, steps)
    x0 = np.concatenate([q_init, p_init], axis=-1)
    print('x0:', x0.shape) # (1, 60)
    x0 = x0.squeeze(0) # (60,)

    def fvec_np(x,t):
        q = torch.tensor(x[:model.n] / q_max, dtype=torch.float64, requires_grad=True).to(device) # (30, )
        p = torch.tensor(x[model.n:] / p_max, dtype=torch.float64, requires_grad=True).to(device) # (30, )
        q = q.view(1, model.n) # (1, 30)
        p = p.view(1, model.n) # (1, 30)

        with torch.enable_grad():
            output = model.time_derivative(q, p, create_graph=False)
        
        dq = output[0].detach().cpu().numpy() * dq_max
        dp = output[1].detach().cpu().numpy() * dp_max
        output = np.concatenate([dq, dp], axis=-1)
        output = output.squeeze(0)    

        return output

    # default (single precision): rtol = atol = 1.49012e-8
    res = scipy.integrate.odeint(fvec_np,x0,t_eval, rtol=1e-12, atol=1e-12)
    return res



def scipy_predict_qp_model_orbits(model, device, q_init, p_init, steps, dt, 
                                  q_max, p_max, dq_max, dp_max):
    model.eval()

    q = torch.tensor(q_init / q_max, dtype=torch.float64, requires_grad=True).to(device)
    p = torch.tensor(p_init / p_max, dtype=torch.float64, requires_grad=True).to(device)

    output = get_orbit (model, device, q_init, p_init, steps, dt, 
                        q_max, p_max, dq_max, dp_max)
    nn_q = output[:, :model.n]
    nn_p = output[:, model.n:]
   
    return nn_q, nn_p



def ground_truth_orbits(q_path, p_path, start_idx, steps):
    q_data = np.load(q_path)['q'].reshape(-1, 30)
    p_data = np.load(p_path)['p'].reshape(-1, 30)
    
    end_idx = start_idx + steps
    
    true_orbit_q = q_data[start_idx:end_idx]
    true_orbit_p = p_data[start_idx:end_idx]
    
    return true_orbit_q, true_orbit_p


def scipy_predict_segment(model, device, gt_q_full, gt_p_full, 
                          start, end, dt,
                          q_max, p_max, dq_max, dp_max):
    model.eval()

    segment_gt_q = gt_q_full[start:end]
    segment_gt_p = gt_p_full[start:end]

    steps = end - start
    q_init = segment_gt_q[0:1]
    p_init = segment_gt_p[0:1]
    segment_nn_q, segment_nn_p = scipy_predict_qp_model_orbits(model, device, q_init, p_init, steps, dt, 
                                                               q_max, p_max, dq_max, dp_max)
    return segment_gt_q, segment_gt_p, segment_nn_q, segment_nn_p
    
def scipy_predict_qp_model_orbits_and_save(model, device, 
                                    q_init, p_init, steps, dt, 
                                    q_max, p_max, dq_max, dp_max,
                                    save_dir=".", save_file_name=None):
    q_pred, p_pred = scipy_predict_qp_model_orbits(model, device, 
                                  q_init, p_init, steps, dt, 
                                  q_max, p_max, dq_max, dp_max)
    if save_file_name is not None:
        save_path = save_dir + "/" + save_file_name                   
        np.savez_compressed(
            save_path,
            q=q_pred,          
            p=p_pred,          
        )





