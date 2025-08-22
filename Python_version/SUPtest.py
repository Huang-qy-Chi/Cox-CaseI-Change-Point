import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
import random
from Lam_estmation import bpoly
from Lam_estmation import Lambda0
from cph import cph 



#%%------------------------------------------SUP test-------------------------------------------
def SUP_test(data, m, K=5, boot = 5000, alpha=0.05, seed = 42):
    Z_2 = data['Z_2']
    Z = data['Z']
    De = data['De']
    n = len(De)
    Z0 = np.ones(n); Z0 = Z0.reshape(-1,1)
    Z1 = np.hstack((Z0,Z))  #Z1=(1,Z), n*(1+p)
    p = Z1.shape[1]-1 #
    step_len = (np.max(Z_2)-np.min(Z_2)-0.1)/K
    zeta_div = np.arange(start=np.min(Z_2)+0.1,stop=np.max(Z_2),step=(np.max(Z_2)-np.min(Z_2))/K)
    def SUP_value(data, zeta_div):
        Z_2 = data['Z_2']
        Z = data['Z']
        De = data['De']
        n = len(De)
        Z0 = np.ones(n); Z0 = Z0.reshape(-1,1)
        Z1 = np.hstack((Z0,Z))  #Z1=(1,Z), n*(1+p)
        p = Z1.shape[1]-1 #
        Res_cox = cph(data, m, graph=False)
        Theta = Res_cox['Theta']
        Lambda_U = Res_cox['Lambda_U']
        r0 = Lambda_U*np.exp(Z@Theta)   #cumulative hazard
        r1 = np.exp(-r0)    # survival
        SUP_0 = []
        for zetao in zeta_div:
            # Z_2_zeta = (Z_2>zetao)[:, np.newaxis] * np.ones((1, p+1), dtype=int)
            # ZC0 = np.hstack((Z,Z1*(Z_2_zeta)))
            h_v = r0  
            Q_y = h_v * (De * np.exp(-h_v)/(1-np.exp(-h_v)+1e-8) - (1-De))*(Z_2>zetao)  #n*1
            Z1_tilde = Z1 
            score1 = (Q_y.T@Z1_tilde).T   #(p+1)*1
            score1 = score1.reshape(-1,1)
            cong = Q_y**2
            score2 = Z1.T@np.diag(cong)@Z1
            SUP_cal = score1.T@np.linalg.inv(score2+1e-8)@score1
            SUP_0.append(SUP_cal)
        SUP_0 = np.array(SUP_0, dtype='float32')
        SUP_stat = np.max(SUP_0)

        return SUP_stat
    
    SUP_statistic = SUP_value(data, zeta_div)
    
    SUP_quantile = []
    key_to_shuffle = 'Z_2'
    for s in range(boot):
        random.seed(seed+s)
        data_copy = data
        random.shuffle(data_copy[key_to_shuffle])
        quantile = SUP_value(data_copy,zeta_div)
        SUP_quantile.append(quantile)

    SUP_quantile = np.array(SUP_quantile)
    SUP_quantile = np.sort(SUP_quantile)
    loc = int(np.floor(boot*(1-alpha)))
    Quan = SUP_quantile[loc]
    index = (SUP_statistic>=Quan).astype(int)
    pquan = np.abs(SUP_quantile-SUP_statistic)
    p_va = 1-np.argmin(pquan)/boot

    return {
        'Decision': index,
        'Statistic': SUP_statistic,
        'Quantile': Quan,
        'P_value':  p_va
    }


