import numpy as np
import pandas as pd
from seed import set_seed
from cpcph import cpcph

#%%-----------------------------------generate bootstrap data-----------------------------------
def bootstrap_dict_mixed(data, n_boot, n_samples=1, random_state=None):
        """
        Bootstrap for dictionary.
        
        Input:
            data (dict): key: U,De,Z,Z_2,T_true(fake), contain n*1 vectors and n*p matrices
            n_boot: sample size of the bootstrap dataset
            n_samples (int): Bootstrap sample's number, initial 1: resample for 1 time
            random_state (int): seed
        
        Output:
            list: list contains n_samples bootstrap sample, each is a dictionary
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # 获取样本数（假设所有键的值有相同的n）
        keys = list(data.keys())
        first_value = np.asarray(data[keys[0]])
        n = first_value.shape[0]
        
        # 检查所有键的值是否有效
        for key in keys:
            value = np.asarray(data[key])
            if value.shape[0] != n:
                raise ValueError(f"Key {key}'s length or column length must be {n}")
            if value.ndim not in [1, 2]:
                raise ValueError(f"Key {key} must be vetor or matrix")
        
        # 初始化结果列表
        bootstrap_samples = []
        
        # 生成m_samples个Bootstrap样本
        for _ in range(n_samples):
            # 有放回随机抽样m个索引
            sampled_indices = np.random.choice(n, size= n_boot, replace=True)
            
            # 构造新字典
            bootstrap_data = {}
            for key in keys:
                value = np.asarray(data[key])
                if value.ndim == 1:
                    # 向量：抽取对应元素
                    bootstrap_data[key] = value[sampled_indices]
                else:
                    # 矩阵：抽取对应行
                    bootstrap_data[key] = value[sampled_indices, :]
            
            bootstrap_samples.append(bootstrap_data)
        
        return bootstrap_samples if n_samples > 1 else bootstrap_samples[0]




#%%-----------------------------------------
def mn_boot(data,q=3,m=3,boot=5000,seed=42,B=100,seq=0.01):
    Res_n = cpcph(data,m=m,B=B,seq=seq)
    zeta_n = Res_n['zeta']
    Z_2 = data['Z_2']
    n = len(Z_2)
    # Step 2: 
    j = np.arange(1, q)
    m_j = np.floor(n * j / q).astype(int)
    x = np.linspace(np.min(Z_2)-5, np.max(Z_2)+5, num=20)
    h = len(x)
    w = len(m_j)
    ecdf = np.zeros((h,w))
    g = 0
    
    for v in m_j:  #different boot sample size
        data_boot = bootstrap_dict_mixed(data,n_boot=v, n_samples=boot, random_state=42)
        zeta_m = []
        for l in range(boot):   #replication 
            set_seed(seed+l)
            data_boot1 = data_boot[l]
            Res_m = cpcph(data_boot1,m=m,B=B,seq=seq)
            zeta_m.append(Res_m['zeta'])
        diff_zeta = zeta_m-zeta_n #boot*1 vector
        cdf = np.zeros((boot,h))
        for s in range(h):
            cdf[:,s] = (v*diff_zeta<x[s]).astype(int)  #boot*h matrix
        ecdf[:,g] = np.mean(cdf,axis=0)  #1*h vector
        g+=1

    dcdf = np.zeros((h,w-1))
    for a in range(w-1):
        dcdf[:,a] = ecdf[:,a+1]-ecdf[:,a] #h*w
    dcdf = np.abs(dcdf)
    diff_cdf = np.zeros(w-1) #w-1 vector
    for r in range(w-1):
        diff_cdf[r] = np.max(dcdf[:,r])
    min_diff_cdf = np.min(diff_cdf)
    min_indices = np.where(diff_cdf == min_diff_cdf)[0]
    # loc = np.argmin(diff_cdf)
    loc = min_indices[-1]
    m1 = n*(1+loc)/q
    m1 = int(m1)
    # print(m)
    return m1

# parallel computation
def mn_boot_par(data,q=3,m=3,boot=5000,seed=42,B=100,seq=0.01,n_jobs=None):
    # U = data['U']; De = data['De'];Z = data['Z'];Z_2 = data['Z_2']
    Res_n = cpcph(data,m=m,B=B,seq=seq)
    zeta_n = Res_n['zeta']
    Z_2 = data['Z_2']
    n = len(Z_2)
    # Step 2: 
    j = np.arange(1, q)
    m_j = np.floor(n * j / q).astype(int)
    x = np.linspace(np.min(Z_2)-5, np.max(Z_2)+5, num=20)
    h = len(x)
    w = len(m_j)
    ecdf = np.zeros((h,w))
    g = 0
    
    for v in m_j:  #different boot sample size
        data_boot = bootstrap_dict_mixed(data,n_boot=v, n_samples=boot, random_state=42)
        
        def parallel_boot(data_boot, key_to_keep='zeta', n_jobs= None):
            """
            data_boot: 包含 B 个重抽样数据集的字典
            n_jobs: 并行进程数（None 表示使用所有可用 CPU 核心）
            返回: 所有回归系数的 NumPy 数组
            """
            if n_jobs is None:
                n_jobs = min(cpu_count() - 1, 60, len(data_boot))   # 
            
            # 获取所有数据集
            datasets = [data_boot[i] for i in range(len(data_boot))]
            
            # 使用 multiprocessing.Pool 并行计算
            # Use "partial" to pass fixed parameters
            sim_func = partial(cpcph, m=m, B=B, seq=seq)
            with Pool(processes=n_jobs) as pool:
                results = pool.map(sim_func, datasets)

            return np.array([result[key_to_keep] for result in results])
        
        # zeta_m = np.array(zeta_m, dtype = 'float32')
        zeta_m = parallel_boot(data_boot, key_to_keep='zeta', n_jobs=n_jobs)


        diff_zeta = zeta_m-zeta_n #boot*1 vector
        cdf = np.zeros((boot,h))
        for s in range(h):
            cdf[:,s] = (v*diff_zeta<x[s]).astype(int)  #boot*h matrix
        ecdf[:,g] = np.mean(cdf,axis=0)  #1*h vector
        g+=1

    dcdf = np.zeros((h,w-1))
    for a in range(w-1):
        dcdf[:,a] = ecdf[:,a+1]-ecdf[:,a] #h*w
    dcdf = np.abs(dcdf)
    diff_cdf = np.zeros(w-1) #w-1 vector
    for r in range(w-1):
        diff_cdf[r] = np.max(dcdf[:,r])
    min_diff_cdf = np.min(diff_cdf)
    min_indices = np.where(diff_cdf == min_diff_cdf)[0]
    # loc = np.argmin(diff_cdf)
    loc = min_indices[-1]
    m1 = n*(1+loc)/q
    m1 = int(m1)
    # print(m)
    return m1


#%%-----------------------------------------------------------------------------
def interval_zeta(data,m1,zeta,m=3,B=5000,seq=0.01,seed=42,alpha=0.05):
    Z_2 = data['Z_2']
    n = len(Z_2)

    # Bootstrap data
    data_boot = bootstrap_dict_mixed(data,n_boot=m1, n_samples=B, random_state=42)
    # Bootstrap zeta estimation: B times
    zeta_m = []
    for l in range(B):   #replication 
        set_seed(seed+l)
        data_boot1 = data_boot[l]
        Res_m = cpcph(data_boot1,m=m,B=B,seq=seq)
        zeta_m.append(Res_m['zeta'])
    
    zeta_m = np.array(zeta_m, dtype = 'float32')
    diff_zeta = np.abs(zeta_m-zeta)
    diff_zeta = np.sort(diff_zeta)
    quantile_alpha = int(np.floor(B*(1-alpha/2))-1)
    inter_length = diff_zeta[quantile_alpha]
    interval = np.array([zeta-m1*inter_length/n, zeta+m1*inter_length/n],dtype='float32')


    return{
        'inter_length': inter_length,
        'interval': interval
    }


# For parallel calculation: bootstrap
def interval_zeta_par(data,m1,zeta,m=3,B=1000,seq=0.01,seed=42,alpha=0.05,n_jobs=None):
    Z_2 = data['Z_2']
    n = len(Z_2)

    # Bootstrap data
    data_boot = bootstrap_dict_mixed(data,n_boot=m1, n_samples=B, random_state=seed)
    # Bootstrap zeta estimation: B times
    def parallel_boot(data_boot, key_to_keep='zeta', n_jobs= None):
        """
        中文注释没空改了(>O<)
        data_boot: 包含 B 个重抽样数据集的字典
        n_jobs: 并行进程数（None 表示使用所有可用 CPU 核心）
        返回: 所有回归系数的 NumPy 数组
        """
        if n_jobs is None:
            n_jobs = min(cpu_count() - 1, 60, len(data_boot))   # 
        
        # 获取所有数据集
        datasets = [data_boot[i] for i in range(len(data_boot))]
        
        # 使用 multiprocessing.Pool 并行计算
        # Use "partial" to pass fixed parameters
        sim_func = partial(cpcph, m=m, B=B, seq=seq)
        with Pool(processes=n_jobs) as pool:
            results = pool.map(sim_func, datasets)

        return np.array([result[key_to_keep] for result in results])
    
    zeta_m = parallel_boot(data_boot, key_to_keep='zeta', n_jobs=n_jobs)
    diff_zeta = np.abs(zeta_m-zeta)
    diff_zeta = np.sort(diff_zeta)
    quantile_alpha = int(np.floor(B*(1-alpha/2))-1)
    inter_length = diff_zeta[quantile_alpha]
    interval = np.array([zeta-m1*inter_length/n, zeta+m1*inter_length/n],dtype='float32')


    return{
        'inter_length': inter_length,
        'interval': interval
    }












