import numpy as np
import matplotlib.pyplot as plt

def f(x):
    if x>0:
        return x
    else:
        return 0 

fv = np.vectorize(f) 

K = 1000 
N = 10000 
mean = 0 
sigma = 1.0 
kappa = 4.0 

ksi = mean + np.sqrt(sigma) * np.random.normal(0,1, N) 
ksj = mean + np.sqrt(sigma) * np.random.normal(0,1, N) 

rates = np.random.normal(10, 1, N) 

print( np.dot(rates, ksi) / N ) 

x = ( 1 + kappa * ksi*ksj / np.sqrt(K) ) 
print('<x>', x.mean() , 'sigma_lim', np.sqrt(K)/16 ) 

print(x[x<0]) 
plt.hist(x)
