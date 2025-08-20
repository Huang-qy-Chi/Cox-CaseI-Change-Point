import numpy as np

#%%---------------------------Least Favourate Direction----------------------------------------
def LFDLCP(data,Lambda_U,Theta,zeta):
    # Step 1: data loading
    # Lambda_U = data['Lambda_U']
    Z = data['Z']
    Z_2 = data['Z_2']
    De = data['De']
    U = data['U']


    n = len(Z_2)
    Z0 = np.ones(n); Z0 = Z0.reshape(-1,1)
    Z1 = np.hstack((Z0,Z))  #Z1=(1,Z)
    p = Z1.shape[1]-1 #
    Z_2_zeta = (Z_2>zeta)[:, np.newaxis] * np.ones((1, p+1), dtype=int)
    ZC0 = np.hstack((Z,Z1*(Z_2_zeta)))
    d = ZC0.shape[1]
    
    # X_U = np.c_[data['X'], data['U']]
    h_v = Lambda_U * np.exp(ZC0@Theta)   #CUMULATIVE HAZARD
    Q_y = h_v * (De * np.exp(-h_v)/(1-np.exp(-h_v)+1e-8) - (1-De))
    S_y = np.exp(-h_v)  #survival

    # X_train1 = np.hstack((np.ones((X_train.shape[0],1)),X_train))
    # n = X_train1.shape[0]
    # d = X_train1.shape[1]
    # ind = (Z_2>zeta)
    # for i in range(d-1):
    #     ind = np.vstack((ind,Z_2>zeta))
    # ind = ind.T
    # ZX = np.hstack((Z_train,X_train1,X_train1*ind))
    ZX = ZC0
    RX = Q_y**2
    LFD = np.mean(np.diag(RX)@ZX, axis = 0)/np.mean(RX)
    #Info = (ZX-LFD).T@np.diag(RX)@(ZX-LFD)
    Info = (ZX-LFD).T@np.diag(RX)@(ZX-LFD)/n
    dinfo = Info.shape[0]
    minvar = np.linalg.inv(Info + np.eye(dinfo)*1e-6)/n
    vars = np.diag(minvar)
    # se1 = np.sqrt(vars[0])
    # se2 = np.sqrt(vars[1])
    se = np.sqrt(vars)

    return {
        'Infobound': minvar,
        'se': se,
        'LFD': LFD,
        'Q_y': Q_y,
    }






























