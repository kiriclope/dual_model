import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi/2, np.pi/2, 100)

def sig(x) :
    return 1/(1+np.exp(-x))

def gauss(x, s):
    return 1/np.sqrt(2*np.pi*s) * np.exp(-x**2 / 2 / s**2)

K = 2000


plt.figure(figsize=(5.663, 3.5))
# plt.plot(x, gauss(x, .25) - gauss(x, 2) , 'b' ) 

plt.plot(x, 1 + 2 * .25 * np.cos(2*x) +  2 * 2.5 * np.cos(4*x)  / np.sqrt(K) , 'b' ) 
plt.plot(x, 5 * (1 + 2 * .25 * np.cos(2*x) + 2 * 2.5 * np.cos(4*x) / np.sqrt(K) ) , 'r' ) 

# plt.plot(x, 1* np.exp(-2*x**2/2) , 'b' ) 
# plt.plot(x, 4 * np.exp(-2*x**2/2 ) , 'r' ) 
plt.savefig('cos_func.svg', dpi=300)
