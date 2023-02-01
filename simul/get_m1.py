import scipy as sp
import numpy as np
import params as gv 
import matplotlib.pyplot as plt

def decode_bump(signal, axis=-1):
    signal_copy = signal.copy()
    if axis!=-1 and signal.ndim!=1 :
        signal_copy = np.swapaxes(signal_copy, axis, -1) 
    
    length = signal_copy.shape[-1]
    dPhi = np.pi/length 
    
    dft = np.dot(signal_copy, np.exp(-2.0j * np.arange(length) * dPhi) )
    
    if axis!=1 and signal.ndim!=1 :
        dft = np.swapaxes(dft, axis, -1) 
        
    m1 = 2 * np.absolute(dft) / length 
    phi = 0.5 * ( np.arctan2(dft.imag, dft.real) % ( 2.0 * np.pi ) ) 
    
    return m1, phi

def circular_convolution(signal, windowSize, axis=-1): 
    signal_copy = signal.copy() 
    
    if axis!=-1 and signal.ndim!=1 : 
        signal_copy = np.swapaxes(signal_copy, axis, -1) 
    
    ker = np.concatenate((np.ones((windowSize, )), np.zeros((signal_copy.shape[-1] - windowSize, )))) 
    smooth_signal = np.real(np.fft.ifft( np.fft.fft(signal_copy, axis=-1) * np.fft.fft(ker, axis=-1) , axis=-1) ) * ( 1.0 / float(windowSize) ) 
    
    if axis!=1 and signal.ndim!=1 : 
        smooth_signal = np.swapaxes(smooth_signal, axis, -1) 
    
    return smooth_signal
    
def get_m1_phi_smooth_rates(rates, n_size=gv.n_size, window=.1): 
    
    m1 = np.zeros( (rates.shape[1], rates.shape[0]) ) * np.nan 
    phi = np.zeros( (rates.shape[1], rates.shape[0]) ) * np.nan 
    smooth_rates = np.zeros( (rates.shape[1], rates.shape[0], rates.shape[-1]) ) * np.nan 
    
    for i_pop in range(rates.shape[1]) : 
        
        pop_rates = rates[:, i_pop, : n_size[i_pop]].copy() 
        pop_rates  =  np.flip(pop_rates, axis=-1) 
        
        m1[i_pop], phi[i_pop] = decode_bump(pop_rates) 
        
        smooth_rates[i_pop, :, :n_size[i_pop]]  = circular_convolution(pop_rates, int(pop_rates.shape[-1] * window) ) 
        
    return m1, phi, smooth_rates 
