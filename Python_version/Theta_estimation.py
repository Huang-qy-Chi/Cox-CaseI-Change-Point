import numpy as np
import scipy.optimize as spo
#estimating the regression parameter
def Theta_est(De, Z, Z_2, Lambda_U, zeta):
    """
    Estimate the parameter: Theta = (beta, intercept, gamma)
    Input: 
    De: 0=fail in [0,U], 1=fail in [U,infty], n*1 vector;
    Z: p-dimensional covariate, n*p matrix; 
    Z_2: change-point covariate, n*1 vector; 
    Lambda_U: baseline cumulative hazard, n*1 vector;
    zeta: estimated change point, a number;
    Output: 
    Theta: estimated parameter Theta, (2*p+1)-dimensional parameter
    """
    n = len(Z_2)
    Z0 = np.ones(n)
    Z0 = Z0.reshape(-1,1)
    Z1 = np.hstack((Z0,Z))  #Z1=(1,Z)
    p = Z1.shape[1]-1 #
    Z_2_zeta = (Z_2>zeta)[:, np.newaxis] * np.ones((1, p+1), dtype=int)
    ZC0 = np.hstack((Z,Z1*(Z_2_zeta)))
    d = ZC0.shape[1]
    def TF(*args):   
        # ZC = np.vstack((Z,Z*(Z_2>zeta)))
        Lam = Lambda_U * np.exp(ZC0 @ args[0])
        Loss_F = np.mean(-De * np.log(1 - np.exp(-Lam) + 1e-5) + (1 - De) * Lam)
        return Loss_F
    result = spo.minimize(TF,np.ones(d),method='SLSQP') #nonconvex optimaization
    Theta_est = result['x']
    Theta_est = Theta_est.astype(np.float32)
    return Theta_est
   



































