import numpy as np
import matplotlib.pyplot as plt

Vth=-50
Vl=-70

mean_V = - np.random.rand() * (Vth-Vl) + Vth
sigma_V =  np.random.rand() * (Vth-Vl) / 4 

print(mean_V, sigma_V)

V = np.random.normal(mean_V, sigma_V, (10000)) 
plt.figure() 
plt.hist(V) 
plt.axvline(Vth, color='k',lw=2) 
plt.axvline(Vl,color='k',lw=2) 
