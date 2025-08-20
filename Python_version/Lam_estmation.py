import numpy as np
import math
import scipy.optimize as spo


#%%-------------------------------Bernstein Polynomial Basises---------------------------------
def bpoly(t,k,m,u,v):
    """
    Calculate the kth Bernstein Polynominal
    Input: 
    t: time (interval censored), n*1 vector
    m: maximum of the number of basis, from 0 to m, (m+1) in total 
    k: the order of basis functon, from 0 to m
    u: max(t) ;  v: min(t0);
    Output: 
    Bernstein Polynomial basis, n*1 vector.
    """
    a = math.comb(m,k)
    b = ((t-v)/(u-v))**k
    c = (1-(t-v)/(u-v))**(m-k)
    return a*b*c



#%%---------------------------------estimate phi para------------------------------------------------
def phi_estmation(U, De, Z, Z_2, Theta, zeta, m): 
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
    Z0 = np.ones(n)
    Z0 = Z0.reshape(-1,1)
    Z1 = np.hstack((Z0,Z))  #Z1=(1,Z)
    Z_2_zeta = (Z_2>zeta)[:, np.newaxis] * np.ones((1, p+1), dtype=int)

    # Step 2. Construct Bernstein Polynominal basis function
    B = np.zeros((n,m+1))  
    for j in range(m+1):
        B[:,j] = bpoly(U, k=j, m=m, u=u, v=v)

    # Step 3. Calculate the parameters phi of basis
    def LF(*args):
        a = args[0]
        a1 = np.cumsum(np.exp(a))
        ZC = np.hstack((Z,Z1*(Z_2_zeta)))
        Lam1 = np.dot(B,a1) * np.exp(ZC@Theta)
        Loss_F1 = np.mean(-De * np.log(1-np.exp(-Lam1)+1e-5) + (1-De)*Lam1)
        return Loss_F1
    # bnds = []
    # for i in range(m+1):
    #     bnds.append((-np.log(1000),np.log(1000)))
    # result = spo.minimize(LF,np.zeros(m+1),method='SLSQP',bounds=bnds)
    result = spo.minimize(LF,np.zeros(m+1),method='BFGS')
    return result['x']


#%%_---------------------------estimate baseline cumulative hazard value-------------------------
def Lambda0(time, phi):
    """
    Calculate the baseline cumulative hazard function
    Input: 
    time: interval censored failure time, n*1 vector
    phi: parameters of the basis, (m+1)*1 vector
    m: maximum of the number of basis, from 0 to m, (m+1) in total 
    u: max(t) ;  v: min(t0);
    Output: 
    Lambda0: the estimated baseline cumulative hazard
    """
    u = np.max(time) #
    v = np.min(time)
    n = len(time)
    phi_s = np.cumsum(np.exp(phi)) 
    m = len(phi)-1
    # Bernstein Polynominal
    B = np.zeros((n,m+1))  
    for j in range(m+1):
        B[:,j] = bpoly(time, k=j, m=m, u=u, v=v)

    Lambda0 = B@phi_s

    return Lambda0












