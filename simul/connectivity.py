import sys, os, importlib 
from importlib import reload
from joblib import Parallel, delayed

import struct 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv 
importlib.reload(sys.modules['params']) 

from utils import open_binary
gv.init_param()

print('reading connectivity from:', gv.con_path) 

figtitle = 'connectivity' 
fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5*4, 1.618*1.25*4)) 

gv.n_neurons = int(gv.n_neurons * 10000)

raw_con_mat = open_binary(gv.con_path, 'con_vec', np.dtype("i4") ) 
raw_con_mat = raw_con_mat.reshape(gv.n_neurons, gv.n_neurons) 

# idx_post = open_binary(gv.con_path, 'idx_post', np.dtype("L") ) 
# n_post = open_binary(gv.con_path, 'idx_post', np.dtype("i4") ) 
# id_post = open_binary(gv.con_path, 'id_post', np.dtype("L") ) 

# im = plt.imshow(raw_con_mat, cmap='jet', vmin=0, vmax=1, origin='lower') 
# plt.grid(False) 
# plt.xlabel('Presynaptic') 
# plt.ylabel('Postsynaptic') 
# plt.title('Connections') 

if(gv.RANK==2) : 
    idx_perm = open_binary(gv.con_path, 'idx_perm', np.dtype("L") ) 
    print('idx_perm', idx_perm.shape, idx_perm[:10], idx_perm[-10:]) 

# n_pre = np.sum(raw_con_mat, axis=0) 
# n_post = np.sum(raw_con_mat, axis=1) 

# print('n_pre', n_pre[:10], n_pre[-10:], np.mean(n_pre) ) 
# print('n_post', n_post[:10], n_post[-10:], np.mean(n_post) ) 

# # con_prob = np.zeros(gv.n_size[0]) 
# # for i in range(gv.n_size): 
# #     con_prob[i] = sum( np.diag(con_mat, -i) ) + sum( np.diag(con_mat, gv.n_size-i) ) 

def parloop(con_mat, i): 
    size = np.max( con_mat.shape ) 
    return (np.sum( np.diag(con_mat, size-i) ) + np.sum( np.diag(con_mat, -i) ) )/ np.min(con_mat.shape) 

n_per_pop = gv.n_size 
if gv.n_pop==2: 
    cum_n_per_pop = [0, n_per_pop[0], n_per_pop[0]+n_per_pop[1]] 
else: 
    cum_n_per_pop = [0, n_per_pop[0]] 
    
print(n_per_pop) 
print(cum_n_per_pop) 

counter=0 

for i in range(2) : # presynaptic 
    for j in range(2) : # postsynaptic 

        con_mat = raw_con_mat[ cum_n_per_pop[i]:cum_n_per_pop[i+1], cum_n_per_pop[j]:cum_n_per_pop[j+1] ]
        
        if(gv.RANK==2):
            print( idx_perm[ cum_n_per_pop[i]:cum_n_per_pop[i+1] ].shape, idx_perm[ cum_n_per_pop[j]:cum_n_per_pop[j+1]].shape) 
        
            idx_i = idx_perm[ cum_n_per_pop[i]:cum_n_per_pop[i+1] ] 
            idx_j = idx_perm[ cum_n_per_pop[j]:cum_n_per_pop[j+1] ]  
            print(idx_i[:10], idx_j[:10]) 
            
            con_mat_perm = raw_con_mat[:, idx_j] 
            con_mat_perm = con_mat_perm[idx_i] 
            
        print( i, j, raw_con_mat.shape, con_mat.shape) 
        
        con_prob = np.array( Parallel(n_jobs=-1)(delayed(parloop)(con_mat, k) for k in range( n_per_pop[j] ) ) )
        
        if(gv.RANK==2):
            con_prob_perm = np.array( Parallel(n_jobs=-1)(delayed(parloop)(con_mat_perm, k) for k in range( n_per_pop[j] ) ) ) 
        
        n_pres = np.sum(con_mat, axis=0) 
        n_post = np.sum(con_mat, axis=1) 
        
        ax = fig.add_subplot(4,4,1+counter) 
        im = ax.imshow(con_mat, cmap='jet', vmin=0, vmax=1, origin='lower') 
        ax.grid(False) 
        plt.xlabel('Presynaptic') 
        plt.ylabel('Postsynaptic') 
        ax.set_title('Connections') 
        
        ax = fig.add_subplot(4, 4, 2+counter) 
        theta = np.linspace(0, np.pi, n_per_pop[j] ) 
        plt.plot(theta[1:], con_prob[1:], color=gv.pal[j])
        plt.xlabel('$\\theta$')
        plt.ylabel('Prob.') 
        plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], 
                   ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{2}$', r'$\pi$'])
        
        # ax = fig.add_subplot(1, 3, 3+counter) 
        # theta = np.linspace(0, np.pi, n_per_pop[j] ) 
        # plt.plot(theta[1:], con_prob[1:], color=gv.pal[j])
        # plt.xlabel('$\phi$')
        # plt.ylabel('Prob.') 
        # plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], 
        #            ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{2}$', r'$\pi$'])
        
        ax = fig.add_subplot(4,4,3+counter) 
        plt.hist(n_pres, histtype='step', color=gv.pal[j]) 
        plt.xlabel('$K_{pres}$') 
        plt.ylabel('Count') 
        
        ax = fig.add_subplot(4,4,4+counter) 
        plt.hist(n_post, histtype='step', color=gv.pal[j]) 
        plt.xlabel('$K_{post}$') 
        plt.ylabel('Count') 
        
        counter +=4 
        
