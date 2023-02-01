import numpy as np
import scipy.stats as stats
import random 
from numpy.random import randint

from joblib import Parallel, delayed, parallel_backend 
import progressbar as pgb 

def shuffle_parloop(X, statfunction): 
    
    np.random.seed(None) 
    
    X_shuffle = X.copy() 
    rng = np.random.default_rng() 
    rng.shuffle(X_shuffle, axis=-1) 
    
    stat_shuffle = statfunction(X_shuffle) 
    
    return stat_shuffle 
    
def shuffle_stat(X, statfunction, n_samples=1000, n_jobs=-10): 
    
    with pgb.tqdm_joblib( pgb.tqdm(desc='shuffle', total=n_samples) ) as progress_bar: 
        stat_shuffle = Parallel(n_jobs=n_jobs)(
            delayed(shuffle_parloop)(X, statfunction) for _ in range(n_samples) 
        ) 
    # gc.collect() 
    
    stat_shuffle = np.asarray(stat_shuffle) 
    
    return stat_shuffle 
