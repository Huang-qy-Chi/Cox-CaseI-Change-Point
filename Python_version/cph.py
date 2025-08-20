import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
from Lam_estmation import bpoly
from Lam_estmation import Lambda0


#%%--------------------------------Cox Model-----------------------------------------
#estimating the regression parameter
def Theta_cox(De, Z, Lambda_U):
    """
    Estimate the parameter: Theta, p-dimensional
    Input: 
    De: 0=fail in [0,U], 1=fail in [U,infty], n*1 vector;
    Z: p-dimensional covariate, n*p matrix; 
    Lambda_U: baseline cumulative hazard, n*1 vector;
    zeta: estimated change point, a number;
    Output: 
    Theta: estimated parameter Theta, (2*p+1)-dimensional parameter
    """
    n = len(De)
    Z0 = np.ones(n)
    # Z0 = Z0.reshape(-1,1)
    # Z1 = np.hstack((Z0,Z))  #Z1=(1,Z)
    # p = Z1.shape[1]-1 #
    # Z_2_zeta = (Z_2>zeta)[:, np.newaxis] * np.ones((1, p+1), dtype=int)
    # ZC0 = np.hstack((Z,Z1*(Z_2_zeta)))
    d = Z.shape[1]
    def TF(*args):   
        # ZC = np.vstack((Z,Z*(Z_2>zeta)))
        Lam = Lambda_U * np.exp(Z @ args[0])
        Loss_F = np.mean(-De * np.log(1 - np.exp(-Lam) + 1e-5) + (1 - De) * Lam)
        return Loss_F
    result = spo.minimize(TF,np.ones(d),method='SLSQP') #nonconvex optimaization
    Theta_est = result['x']
    Theta_est = Theta_est.astype(np.float32)
    return Theta_est



def phi_cox(U, De, Z, Theta, m): 
    """
    Estimate parameters of the Bernstein Polynomial basises:  
    Input:
    U: interval censored failure time; 
    De: censoring indicator: 0=fail at [0,U], 1=fail at [U,infty];
    Z: p-dimensional 
    Output:
    (m+1)-dimensional basises parameter phi
    """

    u = np.max(U)
    v = np.min(U)
    n = len(U)
    p = int((len(Theta)-1)/2)
    
    
    # Step 1. With intercept in change point
    # Z0 = np.ones(n)
    # Z0 = Z0.reshape(-1,1)
    # Z1 = np.hstack((Z0,Z))  #Z1=(1,Z)
    # Z_2_zeta = (Z_2>zeta)[:, np.newaxis] * np.ones((1, p+1), dtype=int)

    # Step 2. Construct Bernstein Polynominal basis function
    B = np.zeros((n,m+1))  
    for j in range(m+1):
        B[:,j] = bpoly(U, k=j, m=m, u=u, v=v)

    # Step 3. Calculate the parameters phi of basis
    def LF(*args):
        a = args[0]
        a1 = np.cumsum(np.exp(a))
        # ZC = np.hstack((Z,Z1*(Z_2_zeta)))
        Lam1 = np.dot(B,a1) * np.exp(Z@Theta)
        Loss_F1 = np.mean(-De * np.log(1-np.exp(-Lam1)+1e-5) + (1-De)*Lam1)
        return Loss_F1
    # bnds = []
    # for i in range(m+1):
    #     bnds.append((-np.log(1000),np.log(1000)))
    # result = spo.minimize(LF,np.zeros(m+1),method='SLSQP',bounds=bnds)
    result = spo.minimize(LF,np.zeros(m+1),method='BFGS')
    return result['x']

#%%------------------Cox model for current status data (Huang 1996, AOS)----------------------------
def cph(data, m, B=100, seq = 0.01, graph = False):

    # Step 1: data_loading
    U = data['U']; De = data['De']; Z = data['Z']
    # T_true = data['T_true']
    n = len(De)
    # zeta0 = np.mean(Z_2) # initial zeta
    # Z0 = np.ones(n); Z0 = Z0.reshape(-1,1)
    # Z1 = np.hstack((Z0,Z))  #Z1=(1,Z)
    # p = Z1.shape[1]-1 #
    # Z_2_zeta = (Z_2>zeta0)[:, np.newaxis] * np.ones((1, p+1), dtype=int)
    # ZC0 = np.hstack((Z,Z1*(Z_2_zeta)))
    d = Z.shape[1]

    # Step 2: initial value setting
    C_index = 0 # whether it converge
    Theta0 = np.ones(d) # initial Theta
    # zeta0 = np.mean(Z_2) # initial zeta
    # Theta0 = np.array([-1,0.5,1.5],dtype='float32')
    # zeta0 = 2

    # Step 3: profile estimation procedure
    for loop in range(B):
        phi = phi_cox(U, De, Z, Theta0, m)
        Lambda_U = Lambda0(U, phi)
        # Lambda_U = U
        Theta1 = Theta_cox(De, Z, Lambda_U)
        # zeta1 = zeta_est(De, Z, Z_2, Lambda_U, Theta0, seq = seq)
        # Theta1 = np.array([-1,0.5,1.5],dtype='float32')
        # zeta1 = 2
        if (np.max(abs(Theta0-Theta1)) <= 0.01):
            C_index = 1
            break
        Theta0 = Theta1
        # zeta0 = zeta1
    ## estimation results
    phi_value = phi; Theta_value = Theta1; 
    # zeta_value = zeta1 
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
        'Theta': Theta_value
        # 'zeta': zeta_value
    }