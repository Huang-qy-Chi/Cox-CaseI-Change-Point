import numpy as np
# import math
# import scipy.optimize as spo
import matplotlib.pyplot as plt
from Lam_estmation import phi_estmation
from Lam_estmation import Lambda0
from Theta_estimation import Theta_est
from zeta_estimation import zeta_est



def cpcph(data, m, B=100, seq = 0.01, graph = False):

    # Step 1: data_loading
    U = data['U']; De = data['De']; Z = data['Z']; Z_2 = data['Z_2']
    # T_true = data['T_true']
    n = len(Z_2)
    zeta0 = np.mean(Z_2) # initial zeta
    Z0 = np.ones(n); Z0 = Z0.reshape(-1,1)
    Z1 = np.hstack((Z0,Z))  #Z1=(1,Z)
    p = Z1.shape[1]-1 #
    Z_2_zeta = (Z_2>zeta0)[:, np.newaxis] * np.ones((1, p+1), dtype=int)
    ZC0 = np.hstack((Z,Z1*(Z_2_zeta)))
    d = ZC0.shape[1]

    # Step 2: initial value setting
    C_index = 0 # whether it converge
    Theta0 = np.ones(d) # initial Theta
    zeta0 = np.mean(Z_2) # initial zeta
    # Theta0 = np.array([-1,0.5,1.5],dtype='float32')
    # zeta0 = 2

    # Step 3: profile estimation procedure
    for loop in range(B):
        phi = phi_estmation(U, De, Z, Z_2, Theta0, zeta0, m)
        Lambda_U = Lambda0(U, phi)
        # Lambda_U = U
        Theta1 = Theta_est(De, Z, Z_2, Lambda_U, zeta0)
        zeta1 = zeta_est(De, Z, Z_2, Lambda_U, Theta0, seq = seq)
        # Theta1 = np.array([-1,0.5,1.5],dtype='float32')
        # zeta1 = 2
        if (np.max(abs(Theta0-Theta1)) <= 0.01):
            C_index = 1
            break
        Theta0 = Theta1
        zeta0 = zeta1
    ## estimation results
    phi_value = phi; Theta_value = Theta1; zeta_value = zeta1 
    Lambda_U_value = Lambda_U

    # (Optional) Step 4: plot the graph of the baseline cumulative hazard
    if graph == True:
        plt.plot(np.sort(U), np.sort(Lambda_U),label=r'Estimated $\Lambda_0$',color='blue',linestyle='--') # Estimated
        plt.plot(np.sort(U),np.sort(U),label=r'True $\Lambda_0$',color='red',linestyle='-')  # True
        plt.gca().set_aspect('equal',adjustable='box')
        plt.grid(True, linestyle = '--',alpha=0.7)
        plt.xlabel('$t$', fontsize=12, color='black')
        plt.ylabel(r'$\Lambda_0(t)$', fontsize=12, color='black')
        plt.legend(fontsize=10)
        # plt.title('Baseline Cumulative Function Graph', fontsize=14)

    return{
        'C_index': C_index,
        'phi': phi_value, 
        'Lambda_U': Lambda_U_value, 
        'Theta': Theta_value, 
        'zeta': zeta_value
    }































