import numpy as np 
from scipy.stats.stats import kendalltau
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import progressbar as pgb 

def perm_corr(a, b):
    '''
    Input:
	two permutations of a set
    Output:
	a value in [-1.0, 1.0] indicating the correlation between the permutations
	a result of 1.0 indicates the permutations are identical
	a result of -1.0 indicates the permutations are reversed
    '''
    assert len(a) == len(b) 
    ssd = 0 # sum of squared differences of element positions
    for pos, elem in enumerate(a): 
        ssd += (pos - np.where(b==elem)[0][0] )**2        
    l = len(a) 
    max_ssd = (l + 1) * l * (l - 1) / 3 # https://oeis.org/A007290 shifted by 1 
    return 1 - 2.0 * ssd / max_ssd 

def unit_vector(vector): 
    """ Returns the unit vector of the vector.  """
    norm = np.linalg.norm(vector, axis=0)
    
    if norm>0: 
        u = vector / norm 
    else: 
        u = np.zeros(vector.shape) 
    
    return u 

def cos_between(v1, v2): 
    """ Returns the angle in radians between vectors 'v1' and 'v2':: """
    
    # idx = np.unique(np.concatenate( ( np.where(v1!=np.nan)[0], np.where(v2!=np.nan)[0]) ) ) 
    
    v1 = v1[idx] 
    v2 = v2[idx] 
    
    v1_u = unit_vector(v1) 
    v2_u = unit_vector(v2) 
    return np.clip( np.dot(v1_u, v2_u), -1.0, 1.0) 

rng = np.random.default_rng() 

n_perm = 10000
n_neurons = 10000 
idx = np.arange(n_neurons)

theta = np.linspace(0, np.pi, n_neurons) 

# corr_list = [] 
# cos_list = [] 
# kendall_list = [] 

# for _ in range(1000): 
#     # idx_perm = rng.permutation(idx.copy()) 
#     rng.shuffle(idx_perm) 
    
#     kendall, pval = kendalltau(idx, idx_perm)  
#     cos = cos_between(idx, idx_perm) 
#     corr = perm_corr(idx, idx_perm) 
    
#     kendall_list.append(kendall) 
#     corr_list.append(corr) 
#     cos_list.append(cos) 
    
#     # print('perm_corr', corr, 'kendall', kendall, 'cos', cos, 'idx', idx[:10], 'idx_perm', idx_perm[:10]) 


def parloop(idx, verbose=0):
    
    idx_perm = idx.copy() 
    rng = np.random.default_rng(None) 
    rng.shuffle(idx_perm) 
    
    kendall, pval = kendalltau(idx, idx_perm) 
    cos = cos_between(idx, idx_perm) 
    corr = perm_corr(idx, idx_perm) 

    if(verbose):
        print('perm_corr', corr, 'kendall', kendall, 'cos', cos, 'idx', idx[:10], 'idx_perm', idx_perm[:10]) 
        
    return corr, cos, kendall

with pgb.tqdm_joblib( pgb.tqdm(desc='generating permutations', total=n_perm ) ) as progress_bar: 
    
    corr_list, cos_list, kendall_list = zip( *Parallel(n_jobs=-2)(delayed(parloop)(idx, verbose=0) 
                                                                  for _ in range(n_perm) ) ) 
    
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.25*1.618*1.5*3, 1.618*1.5)) 
    
ax1.hist(corr_list)
ax1.set_xlabel('corr') 
ax1.set_ylabel('Count') 

ax2.hist(cos_list)
ax2.set_xlabel('cos') 
ax2.set_ylabel('Count') 

ax3.hist(kendall_list) 
ax3.set_xlabel('kendall') 
ax3.set_ylabel('Count') 
