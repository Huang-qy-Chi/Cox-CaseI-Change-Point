import numpy as np

def zeta_est(De, Z, Z_2, Lambda_U,Theta, seq = 0.01):
    """
    Estimate the change-point zeta
    Input: 
    De: 0=fail in [0,U], 1=fail in [U,infty], n*1 vector;
    Z: p-dimensional covariate, n*p matrix; 
    Z_2: change-point covariate, n*1 vector; 
    Lambda_U: baseline cumulative hazard, n*1 vector;
    Theta: estimated regression parameter, (2*p+1) dimensional vector;
    seq: accuracy of gri search, initial = 0.01
    Output: 
    zeta_est: estimation of the change-point by grid search, a number of float32
    """
    #establish the grid of zeta
    Z_min = np.min(Z_2)
    Z_max = np.max(Z_2)
    zeta_grid = np.arange(Z_min, Z_max, seq)  #the search grid
    
    #define the log-likelihood loss
    def BZ2(*args):
        n = len(Z_2)
        Z0 = np.ones(n)
        Z0 = Z0.reshape(-1,1)
        Z1 = np.hstack((Z0,Z))  #Z1=(1,Z)
        p = Z1.shape[1]-1 #
        Z_2_zeta = (Z_2>args[0])[:, np.newaxis] * np.ones((1, p+1), dtype=int)
        ZC0 = np.hstack((Z,Z1*(Z_2_zeta)))
        Lam = Lambda_U * np.exp(ZC0 @ Theta )
        Loss_F = np.mean(-De * np.log(1 - np.exp(-Lam) + 1e-5) + (1 - De) * Lam)
        return Loss_F
    
    zeta_loss = []
    for zeta in zeta_grid:
        zeta_loss.append(BZ2(zeta))
    loss_min = min(zeta_loss)
    loc = zeta_loss.index(loss_min)
    zeta_est = zeta_grid[loc]
    zeta_est = zeta_est.astype(np.float32)
    return zeta_est
        











































