import numpy as np
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from functools import partial
import time
from data_generator import gendata_linear
from SUPtest import SUP_test
from data_generator import search_c

# set random seed to ensure the experiment reproducible
np.random.seed(42)

# Function to generate simulated data and obtain the SUPk test
def SUP_simulation(seed,Theta,zeta,n,m,c,alpha=0.05,corr=0.5,K=5,boot=1000,n_sim=1000):
    """
    Test the existence of change-point

    Input: 
        seed: random seed, ensures the independence of each process
        Theta: true value of Theta = (beta, 1, gamma)
        zeta: true value of the change-point zeta
        n: sample size for each replication
        m: number of the Bernstein Polynominal basis (use (m+1) basises)
        c: value to control the left/right censoring rate in current status data
        alpha: significance level, initial 0.05
        corr: correlation between different covariates, initial 0.5
        K: k in SUPk, number of the splitting times for searching the change-point, initial 5
        boot: number of the bootstrap time, initial 1000
        n_sim: number of replication in one process, initial 1000

    Output: 
        decision: n_sim*1 vector, decision results in one single process
    """
    # set independent seed for each process
    np.random.seed(seed)
    
    decision = np.zeros(n_sim)
    for i in range(n_sim):
        # generate simulated data
        train_data = gendata_linear(n,c,Theta,zeta,corr)
        
        # obtain the SUPk test with significance level alpha and split time K
        decision[i] = SUP_test(train_data, m, K=K, boot = boot, alpha=alpha, seed = seed+i)['Decision']
    
    return decision

# Function for parallel calculation

def parallel_SUP(Theta,zeta,m,n,c,n_sim_total=5000,boot=5000,K=5,alpha=0.05,corr=0.5,seed=42):
    """
    Obtain the SUPk test with multi-core parallel processing

    Input: 
        Theta: true value of Theta = (beta, 1, gamma)
        zeta: true value of the change-point zeta
        m: number of the Bernstein Polynominal basis (use (m+1) basises)
        n: sample size for each replication
        c: value to control the left/right censoring rate in current status data
        n_sim_total: total number of the repulication, initial 5000
        boot: number of the bootstrap time, initial 5000
        K: k in SUPk, number of the splitting times for searching the change-point, initial 5
        alpha: significance level, initial 0.05
        corr: correlation between different covariates, initial 0.5
        seed: parameter to adjust the random seeds, initial 42
       
    Output: 
        SUP_mean: empirical probability that the test reject H0 (no change point), P(SUPk reject H0)
        SUP_decision: empirical results of the decision (0=accept H0, 1=reject H0)
    """
    
    # Retrieve the number of CPU cores
    num_cores = cpu_count()-4  # 4 cores are reserved to balance the computer
    print(f"Use {num_cores} CPU cores.")
    
    # Distribute the total number of simulations to each process
    n_sim_per_core = n_sim_total // num_cores
    seeds = range(seed,seed+num_cores,1)  # Assign different seeds to each process
    
    # Create a process pool
    start_time = time.time()
    with Pool(processes=num_cores) as pool:
        # Use "partial" to pass fixed parameters
        sim_func = partial(SUP_simulation, Theta = Theta, zeta = zeta, n=n, \
                            m=m, c = c, alpha=alpha,corr=corr, K=K, boot=boot, n_sim=n_sim_per_core)
        # Parallel execution, passing in different seeds
        results = pool.map(sim_func, seeds)
    
    # Merge all processes results
    SUP_decision = np.concatenate(results)
    
    # calculate type I error & power
    SUP_mean = np.mean(SUP_decision)

    
    end_time = time.time()
    print(f"Parallel computing time: {end_time - start_time:.2f} seconds for SUP test.")
    
    return SUP_mean, SUP_decision



# An example
if __name__ == '__main__':
    # parameter settings
    n_sim_total = 100  # replication, adjust to 5000
    boot = 500         # Bootstrap time. adjust to 5000
    n = 300              # sample size
    Theta =  np.array([-1,0,0],dtype='float32')  # true parameter
    zeta = 2          # true change point
    m = 4
    U_max = 5
    pr = 0.5  # actully left censoring rate
    K = 5
    alpha = 0.05
    corr = 0.5
    sed = 42
    c = search_c(pr, Theta, zeta, U_max, stepsize=0.01, B=5000, \
                 corr=0.5, seed=sed, graph=False)['c']

    # Execute parallel computing
    SUP_value, SUP_decision = parallel_SUP(Theta,zeta,m,n,c,n_sim_total=n_sim_total,\
                                           boot=boot,K=K,alpha=alpha,corr=corr,seed=sed)
    
    print(f"type I or power: {SUP_value:.2f}")
    print(f"decision: {SUP_decision}")

# Update at 26/08/2025, produced by Qiyue Huang, Hong Kong, China. 