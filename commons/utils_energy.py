
import numpy as np
import torch

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

def ground_truth_kinetic_energy(p):
    # Kinetic Energy: T = 0.5 * m * v^2, where v = p/m
    # T = 0.5 * m * (p/m)^2 = p^2 / (2m)
    return np.sum((np.linalg.norm(p, axis=2)**2) / (2 * mass), axis=1)

def ground_truth_potential_energy(q):
    # Potential Energy: V = -G * sum_{i<j} (m_i * m_j / r_ij)
    V = 0  
    for i in range(mass.shape[0] - 1):
        for j in range(i + 1, mass.shape[0]):
            Rij = np.linalg.norm(q[:, i, :] - q[:, j, :], axis=1)    
            V += -G_cnst * mass[i] * mass[j] / Rij   
                          
    return V

def ground_truth_hamiltonian(q, p):
    # Kinetic Energy
    T = ground_truth_kinetic_energy(p)
    # Potential Energy
    V = ground_truth_potential_energy(q)
    return T + V

def calculate_gt_energy_terms(q_orbit, p_orbit):
    q = q_orbit.reshape(-1, 10, 3)
    p = p_orbit.reshape(-1, 10, 3)
    T = ground_truth_kinetic_energy(p)
    V = ground_truth_potential_energy(q)
    H = T + V
    return T.flatten(), V.flatten(), H.flatten()

def calculate_gt_hamiltonian(q_orbit, p_orbit):
    q = q_orbit.reshape(-1, 10, 3)
    p = p_orbit.reshape(-1, 10, 3)
    H = ground_truth_hamiltonian(q, p)
    return H.flatten()

def calculate_nn_energy_terms(model, device, q_orbit, p_orbit, q_max, p_max, qp_cat=True):
    H_values = []
    T_values = []
    V_values = []

    # q_orbit, p_orbit: (Time, Dim)
    for q, p in zip(q_orbit, p_orbit):
        q_norm = torch.tensor(q / q_max, dtype=torch.float64).to(device)
        p_norm = torch.tensor(p / p_max, dtype=torch.float64).to(device)
        
        if qp_cat:
            # Basic HNN
            x = torch.cat([q_norm, p_norm], dim=-1).unsqueeze(0)
            with torch.no_grad():
                H = model(x)
                T, V = model.energy_terms()
        else:
            # Plausible HNN: (batch, 30)
            qn = q_norm.unsqueeze(0)
            pn = p_norm.unsqueeze(0)
            with torch.no_grad():
                # Plausible HNN: forward(q, p)
                H = model(qn, pn)
                T, V = model.energy_terms()
        
        T_value = T.detach().cpu().item()
        V_value = V.detach().cpu().item()
        H_value = H.detach().cpu().item()
        T_values.append(T_value)
        V_values.append(V_value)
        H_values.append(H_value)

    return np.array(T_values), np.array(V_values), np.array(H_values)


def calculate_nn_hamiltonian(model, device, q_orbit, p_orbit, q_max, p_max, qp_cat=True):
    H_values = []
    # q_orbit, p_orbit: (Time, Dim)
    for q, p in zip(q_orbit, p_orbit):
        q_norm = torch.tensor(q / q_max, dtype=torch.float64).to(device)
        p_norm = torch.tensor(p / p_max, dtype=torch.float64).to(device)
        
        if qp_cat:
            # Basic HNN
            x = torch.cat([q_norm, p_norm], dim=-1).unsqueeze(0)
            with torch.no_grad():
                H = model(x)
        else:
            # Plausible HNN: (batch, 30)
            qn = q_norm.unsqueeze(0)
            pn = p_norm.unsqueeze(0)
            with torch.no_grad():
                # Plausible HNN: forward(q, p)
                H = model(qn, pn)
        
        H_values.append(H.detach().cpu().item())
            
    return np.array(H_values)
    
