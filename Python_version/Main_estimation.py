import numpy as np
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from functools import partial
import time
import scipy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from data_generator import gendata_linear
from data_generator import search_c
from Lam_estmation import Lambda0
from cpcph import cpcph 
from LFD_LCP import LFDLCP
from seed import set_seed
from mn_boot import mn_boot
from mn_boot import interval_zeta


def CPCPH(seed,Theta, zeta, n, m, c, m1, alpha=0.05, corr=0.5, boot=1000,\
           n_sim=1000,B=100,seq=0.01):
    """
    Generate current status data and estimate the Cox model with a change-point

    Input: 
        seed: random seed, ensuring each process's random numbers are independent 
                and reproducible
        Theta: true value of Theta = (beta, 1, gamma)
        zeta: true value of the change-point zeta
        n: sample size for each replication
        m: number of the Bernstein Polynominal basis (use (m+1) basises)
        c: value to control the left/right censoring rate in current status data
        m1: sample size selected from the "m out of n bootstrap" to estimated zeta's 
                confidence interval (CI)
        alpha: significance level for Theta's CP and zeta's CI, initial 0.05
        corr: correlation between different covariates, initial 0.5
        boot: number of the bootstrap time, initial 1000
        n_sim: number of replication in one process, initial 1000
        B: max loop times for iteration, initial 100
        seq: precision of the grid search, initial 0.01 

    Output: 
        Theta_res: estimated Theta values (parameter), n_sim*d matrix
        zeta_res: estimated zeta values (change-point), n_sim*1 vector
        phi_res: estimated parameter of Berstein Polynominal basises 
                (baseline cumulative hazard), n_sim*(m+1) matrix
        se_res: estimated standard error of Theta, n_sim*d matrix 
        zeta_len: estimated length of zeta's CI, n_sim*1 vector
    """
    # set seeds for each pool
    np.random.seed(seed)
    
    zeta_record = np.zeros(n_sim)   # n_sim vector
    zeta_len = np.zeros(n_sim)   # n_sim vector
    d = len(Theta)
    p = int((d-1)/2)
    Theta_record = np.zeros((n_sim, d))   # n_sim*d matrix
    phi_record = np.zeros((n_sim,(m+1)))  #n_sim*(m+1) matrix
    se_record = np.zeros((n_sim, d))   # n_sim*d matrix
    for i in range(n_sim):
        # generate data
        train_data = gendata_linear(n,c,Theta,zeta,corr)
        
        # calculate the SMLE 
        Res = cpcph(train_data, m, B, seq, graph=False)
        zeta_record[i] = Res['zeta']
        Theta_record[i,:] = Res['Theta']
        phi_record[i,:] = Res['phi']
        Lambda_U = Res['Lambda_U']
        se_record[i,:] = LFDLCP(train_data,Lambda_U,Theta,zeta)['se']
        zeta_len[i] = interval_zeta(train_data,m1,zeta,m=m,B=boot,seq=seq,seed=seed,alpha=alpha)['inter_length']
  
    return {
        'Theta_res': Theta_record, 
        'zeta_res': zeta_record, 
        'phi_res': phi_record, 
        'se_res': se_record, 
        'zeta_len': zeta_len
    }


def parallel_CPCPH(Theta,zeta,m,n,c,m1,n_sim_total=1000, boot=1000,\
                    alpha=0.05,corr=0.5, graph=False,B=100,seq=0.01):
    """
    The function estimated  
        a). Bias, SSE, ESE and CP for Theta, 
        b). Bias, Std, length of CI and CI coverage of zeta
    via multi-core parallel processing. If graph==True, plot Lambda0 on the test data. 

    Input: 
        Theta: true value of Theta = (beta, 1, gamma)
        zeta: true value of the change-point zeta
        m: number of the Bernstein Polynominal basis (use (m+1) basises)
        n: sample size for each replication
        c: value to control the left/right censoring rate in current status data
        n_sim_total: total replication times, initial 1000
        boot: number of the bootstrap time, initial 1000
        alpha: significance level, initial 0.05
        corr: correlation between different covariates, initial 0.5
        graph: whether to plot Lambda_0, initial == False
        B: max loop times for iteration, initial 100
        seq: precision of the grid search, initial 0.01 

    Output: 
        All outputs are dictionary with keys 'Theta' and 'zeta'. 
        bias: bias of Theta and zeta (Bias): d*1 vector and a number
        sse: simulated standard error (SSE): d*1 vector and a number
        ese: estimated standard error (ESE): d*1 vector and a number
        cp: 1-alpha coverage probability (CP): d*1 vector and a number
    """
    

    # Retrieve the number of CPU cores
    num_cores = cpu_count() - 4
    print(f"Use {num_cores} CPU cores.")
    # Distribute the total number of simulations to each process
    n_sim_per_core = n_sim_total // num_cores
    seeds = range(num_cores)  # Assign different seeds to each process

    # Create a process pool
    start_time = time.time()
    with Pool(processes=num_cores) as pool:
        # Use "partial" to pass fixed parameters
        sim_func = partial(CPCPH, Theta = Theta, zeta = zeta, n = n, \
                            m = m, c = c, m1 = m1, alpha = alpha, corr = corr,\
                                   boot = boot, n_sim = n_sim_per_core,B=B,seq=seq)
        # Parallel execution, passing in different seeds
        results = pool.map(sim_func, seeds)
        pool.close()
        pool.join()
    # Merge all processes results
    Theta_res = np.concatenate([r['Theta_res'] for r in results], axis=0) # n_sim*d matrix
    zeta_res = np.concatenate([r['zeta_res'] for r in results]) # n_sim vector
    phi_res = np.concatenate([r['phi_res'] for r in results], axis=0) # n_sim*(m+1) vector
    se_res = np.concatenate([r['se_res'] for r in results], axis=0) # n_sim*d matrix
    zeta_len= np.concatenate([r['zeta_len'] for r in results]) # n_sim vector

    # calculate type I error & power
    bias = {
        'Theta': np.mean(Theta_res, axis=0) - Theta,
        'zeta': np.mean(zeta_res) - zeta
    }
    sse = {
        'Theta': np.std(Theta_res, axis=0, ddof=1),
        'zeta': np.std(zeta_res, ddof=1),
    }
    ese = {
        'Theta': np.mean(se_res, axis=0),
        'zeta': np.mean(zeta_len)
    }

    def CP_cal(Theta_res, se_res, Theta, alpha=0.05):
        """
        Calculate the coverage probability (95% confidence interval) of the d-dimensional Theta.
        Parameters:
            Theta_res: (n_sim, d) matrix, estimated Theta values
            se_res: (n_sim, d) matrix, bTheta's se values
            Theta: (d,) vector, true value of Theta
            alpha: significance level, initial 0.05
        Output: 
            cp: (d,) vector, 1-alpha coverage probability
        """
        # significance value towards alpha
        z =  scipy.stats.norm.ppf(1 - alpha/2)
        
        # siginificance interval bounds
        n_sim = Theta_res.shape[0]
        lower_bound = Theta_res - z * se_res  # (n_sim, d)
        upper_bound = Theta_res + z * se_res  # (n_sim, d)
        
        # whether Theta in [lower_bound, upper_bound]
        in_ci = (lower_bound <= Theta) & (Theta <= upper_bound)  # (n_sim, d)
        
        # calculate CP for each dimension
        cp = np.mean(in_ci, axis=0)  # (d,) vector CP
        
        return cp
    
    def CI_cal(zeta_res, zeta_len, zeta, m1):
        
        n_sim = Theta_res.shape[0]
        z = m1/n_sim
        lower_bound = zeta_res - z * zeta_len  # (n_sim, )
        upper_bound = zeta_res + z * zeta_len  # (n_sim, )
        
        # whether zeta in [lower_bound, upper_bound]
        in_ci = (lower_bound <= zeta) & (zeta <= upper_bound)  # (n_sim,)
        
        # calculate CP for each dimension
        ci = np.mean(in_ci)  # CI value

        return ci


    cp = {
        'Theta': CP_cal(Theta_res,se_res,Theta,alpha=alpha),
        'zeta': CI_cal(zeta_res, zeta_len, zeta, m1)
    }
    
    # Plot the Lambda_0 baseline cumulative hazard
    if graph==True:
        test_data = gendata_linear(500,c,Theta,zeta,corr)
        U_test = test_data['U']
        phi_mean = np.mean(phi_res, axis=0)
        Lambda_U_test = Lambda0(U_test, phi_mean)
        plt.plot(np.sort(U_test), np.sort(Lambda_U_test),label=r'Estimated $\Lambda_0$',color='blue',linestyle='--') # Estimated
        plt.plot(np.sort(U_test),np.sort(U_test),label=r'True $\Lambda_0$',color='red',linestyle='-')  # True
        plt.gca().set_aspect('equal',adjustable='box')
        plt.grid(True, linestyle = '--',alpha=0.7)
        plt.xlabel('$t$', fontsize=12, color='black')
        plt.ylabel(r'$\Lambda_0(t)$', fontsize=12, color='black')
        plt.legend(fontsize=10)

    end_time = time.time()
    print(f"Parallel computing time: {end_time - start_time:.2f} seconds for estimation.")

    return {
        'bias': bias, 
        'sse': sse, 
        'ese': ese, 
        'cp': cp
            }


# An example
if __name__ == '__main__':
     # parameter for simulation
    sed = 42
    n_sim_total = 100  # replication time, adjust to 5000
    boot = 100         #Bootstrap time, adjust to 5000
    B = 100            #max iteration time
    n = 300              # sample size
    Theta =  np.array([-1,0.5,1.5],dtype='float32')  # true parameter
    zeta = 2          # true change point
    m = 4
    U_max = 5
    pr = 0.5  # actully left censoring rate
    alpha = 0.05
    seq = 0.01; corr =0.5

    start_time1 = time.time()
    c = search_c(pr,Theta,zeta,U_max,stepsize=seq,B=boot,corr=corr,seed=sed,graph=False)['c']
    print(c)
    set_seed(sed)
    data = gendata_linear(n,c,Theta,zeta,corr=corr) 
    q = 5
    m1 = mn_boot(data,q,m,boot=boot,seed=sed,B=B,seq=seq)   # attention: adjust boot to 5000!
    print(m1)
    end_time1 = time.time()
    print(f"Parameter computing time: {end_time1 - start_time1:.2f} seconds.")
    result = parallel_CPCPH(Theta,zeta,m,n,c,m1,n_sim_total,boot,\
                    alpha,corr, graph=False,B=B,seq=seq)
    
    print("Bias: ")
    for param, value in result['bias'].items():
        print(f"  {param}: {value}")
    print("SSE: ")
    for param, value in result['sse'].items():
        print(f"  {param}: {value}")
    print("ESE: ")
    for param, value in result['ese'].items():
        print(f"  {param}: {value}")
    print("CP: ")
    for param, value in result['cp'].items():
        print(f"  {param}: {value}")

# Update at 26/08/2025, produced by Qiyue Huang, Hong Kong, China. 
