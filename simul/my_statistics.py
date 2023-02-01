import numpy as np
from numpy.random import randint

import scipy.stats as stats
import random 

from joblib import Parallel, delayed, parallel_backend 
import progressbar as pgb 

def shuffle_parloop(X, statfunction): 
    
    np.random.seed(None) 
    
    X_shuffle = X.copy() 
    np.random.shuffle(X_shuffle) 
    stat_shuffle = statfunction(X) 
    
    return stat_shuffle 
    
def shuffle_stat(X, statfunction, n_samples=1000, n_jobs=-10): 
    
    with pgb.tqdm_joblib( pgb.tqdm(desc='shuffle', total=n_samples) ) as progress_bar: 
        stat_shuffle = Parallel(n_jobs=n_jobs)(
            delayed(shuffle_parloop)(X, statfunction) for _ in range(n_samples) 
        ) 
    
    # gc.collect() 
    
    stat_shuffle = np.asarray(stat_shuffle) 
    
    return stat_shuffle 

def bootstrap_parloop(X, statfunction):
    np.random.seed(None) 
    # Sample (with replacement) from the given dataset     
    idx = np.random.choice(X.shape[0], X.shape) 
    stats = statfunction(X[idx]) 
    
    return stats 

def my_bootstraped_ci(X, confidence=0.95, n_samples=1000, statfunction=np.mean, n_jobs=-10): 
    
    with pgb.tqdm_joblib( pgb.tqdm(desc='bootstrap', total=n_samples) ) as progress_bar:
        with parallel_backend("loky", inner_max_num_threads=1):        
            stats = Parallel(n_jobs=n_jobs)(
                delayed(bootstrap_parloop)(X, statfunction) for _ in range(n_samples) 
            ) 
            
    stats = np.asarray(stats) 
    print('stats', stats.shape) 
    
    # Sort the array of per-sample statistics and cut off ends 
    ostats = np.sort(stats, axis=0) 
    mean = np.mean(ostats, axis=0)
        
    p = (1.0 - confidence) / 2.0 * 100 
    lperc = np.percentile(ostats, p , axis=0)
    dum = np.vstack( ( np.zeros(lperc.shape), lperc) ).T 
    
    # lval = mean - lperc 
    lval = lperc 
    
    p = (confidence + (1.0 - confidence) / 2.0 ) * 100 
    uperc = np.percentile(ostats, p , axis=0) 
    dum = np.vstack( ( np.ones(uperc.shape), uperc) ).T 
    
    # uval = -mean + uperc
    uval = uperc
    
    ci = np.vstack((lval, uval)).T 
    
    return mean, ci 
