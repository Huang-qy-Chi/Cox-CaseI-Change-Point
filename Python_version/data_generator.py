import numpy as np
from seed import set_seed
import matplotlib.pyplot as plt

#%%-----------------------Linear Cox model with current status data---------------------------
def search_c(pr, Theta, zeta, U_max, stepsize=0.01, B=1000, corr=0.5, \
              seed=42, graph=False):
    set_seed(seed)
    p = int((len(Theta)-1)/2)
    mean = np.zeros(p)
    cov = np.identity(p)*(1-corr) + np.ones((p, p))*corr
    Z_2 = np.random.normal(loc=zeta,scale=zeta-0.5,size=B)
    Z_2 = np.clip(Z_2,1,3)
    Z = np.random.multivariate_normal(mean,cov,B)
    Z = np.clip(Z, a_min=-1, a_max = 1)
    Z0 = np.ones(B)
    Z0 = Z0.reshape(-1,1)
    Z1 = np.hstack((Z0,Z))  #Z1=(1,Z)
    Z_2_zeta = (Z_2>zeta)[:, np.newaxis] * np.ones((1, p+1), dtype=int)
    ZC = np.hstack((Z,Z1*(Z_2_zeta)))
    u = np.random.uniform(low=0,high=1,size=B)
    T = -np.log(u)/np.exp(ZC@Theta)   #dim B
    c_select = np.arange(0, U_max, stepsize)
    Cenrate = []
    for c_s in c_select:
        C_vec = np.random.uniform(low=0,high=c_s,size=B)  #dim B
        Cenrate.append(np.mean(T<=C_vec))
    loc = np.argmin(np.abs(np.array(Cenrate)-pr))
    c = c_select[loc]
    if graph==True:
        plt.plot(c_select,np.array(Cenrate))
        plt.axhline(y=pr, color='red', linestyle='--', linewidth=2, label='Expected Rate')
        plt.xlabel('c', fontsize=12, color='black')
        plt.ylabel('Right censoring rate', fontsize=12, color='black')
        plt.title('Choice of c', fontsize=14)
    return {
        'c': np.array(c, dtype = 'float32'),
        'rightcen_list': np.array(Cenrate, dtype = 'float32')
    }


def search_c_cox(pr, Theta, U_max, stepsize=0.01, B=1000, corr=0.5, \
              seed=42, graph=False):
    set_seed(seed)
    p = int(len(Theta))
    mean = np.zeros(p)
    cov = np.identity(p)*(1-corr) + np.ones((p, p))*corr
    # Z_2 = np.random.normal(loc=zeta,scale=zeta-0.5,size=B)
    # Z_2 = np.clip(Z_2,1,3)
    Z = np.random.multivariate_normal(mean,cov,B)
    Z = np.clip(Z, a_min=-1, a_max = 1)
    # Z0 = np.ones(B)
    # Z0 = Z0.reshape(-1,1)
    # Z1 = np.hstack((Z0,Z))  #Z1=(1,Z)
    # Z_2_zeta = (Z_2>zeta)[:, np.newaxis] * np.ones((1, p+1), dtype=int)
    # ZC = np.hstack((Z,Z1*(Z_2_zeta)))
    u = np.random.uniform(low=0,high=1,size=B)
    T = -np.log(u)/np.exp(Z@Theta)   #dim B
    c_select = np.arange(0, U_max, stepsize)
    Cenrate = []
    for c_s in c_select:
        C_vec = np.random.uniform(low=0,high=c_s,size=B)  #dim B
        Cenrate.append(np.mean(T<=C_vec))
    loc = np.argmin(np.abs(np.array(Cenrate)-pr))
    c = c_select[loc]
    if graph==True:
        plt.plot(c_select,np.array(Cenrate))
        plt.axhline(y=pr, color='red', linestyle='--', linewidth=2, label='Expected Rate')
        plt.xlabel('c', fontsize=12, color='black')
        plt.ylabel('Right censoring rate', fontsize=12, color='black')
        plt.title('Choice of c', fontsize=14)
    return {
        'c': np.array(c, dtype = 'float32'),
        'rightcen_list': np.array(Cenrate, dtype = 'float32')
    }



#%%-----------------------------Data generator-----------------------------
def gendata_linear(n,c,Theta,zeta,corr=0.5):
    p = int((len(Theta)-1)/2)
    Z_2 = np.random.normal(loc=zeta,scale=zeta-0.5,size=n)
    Z_2 = np.clip(Z_2,1,3)
    mean = np.zeros(p)
    cov = np.identity(p)*(1-corr) + np.ones((p, p))*corr
    Z = np.random.multivariate_normal(mean,cov,n)
    Z = np.clip(Z, a_min=-1, a_max = 1)
    Z0 = np.ones(n)
    Z0 = Z0.reshape(-1,1)
    Z1 = np.hstack((Z0,Z))  #Z1=(1,Z)
    Z_2_zeta = (Z_2>zeta)[:, np.newaxis] * np.ones((1, p+1), dtype=int)
    ZC = np.hstack((Z,Z1*Z_2_zeta))  #2p+1 dim, along with Theta
    u1 = np.random.uniform(low=0,high=c,size=n)
    u0 = np.random.uniform(low=0,high=1,size=n)
    Trec = -np.log(u0)/(np.exp(ZC@Theta))
    De = (Trec<=u1).astype(int)
    U = u1

    return{
        'U': np.array(U, dtype='float32'),
        'De': np.array(De, dtype='float32'), 
        'Z': np.array(Z, dtype='float32'),
        'Z_2': np.array(Z_2, dtype='float32'),
        'T_true': np.array(Trec, dtype='float32'),
    }


#%%-----------------------------Data generator-----------------------------
def gendata_linear_cox(n,c,Theta,corr=0.5):
    p = int(len(Theta))
    # Z_2 = np.random.normal(loc=zeta,scale=zeta-0.5,size=n)
    # Z_2 = np.clip(Z_2,1,3)
    mean = np.zeros(p)
    cov = np.identity(p)*(1-corr) + np.ones((p, p))*corr
    Z = np.random.multivariate_normal(mean,cov,n)
    Z = np.clip(Z, a_min=-1, a_max = 1)
    # Z0 = np.ones(n)
    # Z0 = Z0.reshape(-1,1)
    # Z1 = np.hstack((Z0,Z))  #Z1=(1,Z)
    # Z_2_zeta = (Z_2>zeta)[:, np.newaxis] * np.ones((1, p+1), dtype=int)
    # ZC = np.hstack((Z,Z1*Z_2_zeta))  #2p+1 dim, along with Theta
    u1 = np.random.uniform(low=0,high=c,size=n)
    u0 = np.random.uniform(low=0,high=1,size=n)
    Trec = -np.log(u0)/(np.exp(Z@Theta))
    De = (Trec<=u1).astype(int)
    U = u1

    return{
        'U': np.array(U, dtype='float32'),
        'De': np.array(De, dtype='float32'), 
        'Z': np.array(Z, dtype='float32'),
        # 'Z_2': np.array(Z_2, dtype='float32'),
        'T_true': np.array(Trec, dtype='float32')
    }

























