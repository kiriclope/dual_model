import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import savgol_filter 
from scipy.stats import spearmanr
from scipy import optimize

from get_m1 import *

def cos_fit_func(x, m0, m1, phi): 
    return m0 + m1 * np.cos(2*x + phi) 

def fft_denoise(raw_signal, percentage): 
    signal = raw_signal - np.mean(raw_signal, axis=0)
    length = signal.shape[0] 
    
    fft_signal = np.fft.fft(signal) 
    fft_abs = np.abs(fft_signal) 
    
    threshold = percentage * ( 2*fft_abs[0:int(length/2.)] /length ) .max() 
    
    fft_th = 2 * fft_abs / length 
    
    fft_signal[fft_th<=threshold]=0 

    return np.real(np.fft.ifft(fft_signal)) 
    
def filter_signal(th):
    f_s = fft_filter(th) 
    return np.real(np.fft.ifft(f_s)) 

# def fft_filter(perc):
#     fft_signal = np.fft.fft(signal) 
#     fft_abs = np.abs(fft_signal) 
#     th=perc*(2*fft_abs[0:int(len(signal)/2.)]/len(new_Xph)).max()
#     fft_tof=fft_signal.copy()
#     fft_tof_abs=np.abs(fft_tof)
#     fft_tof_abs=2*fft_tof_abs/len(new_Xph)
#     fft_tof[fft_tof_abs<=th]=0 
#     return fft_tof 

def fft_filter_amp(th):
    fft = np.fft.fft(signal)
    fft_tof=fft.copy()
    fft_tof_abs=np.abs(fft_tof)
    fft_tof_abs=2*fft_tof_abs/len(new_Xph)
    fft_tof_abs[fft_tof_abs<=th]=0
    return fft_tof_abs[0:int(len(fft_tof_abs)/2.)]

def corr_signal_filter():
    th_list = np.linspace(0, 1, 1000) 
    th_list = th_list[0:len(th_list)] 
    p_values = []
    corr_values = []
    for t in th_list:
        filt_signal = filter_signal(t) 
        res = spearmanr(signal, signal-filt_signal) 
        p_values.append(res.pvalue)
        corr_values.append(res.correlation)

def find_threshold(signal):

    th_list = np.linspace(0, 1, 1000) 
    corr_values = [] 
    for threshold in th_list: 
        filt_signal = fft_denoise(signal, threshold) 
        res = spearmanr(signal,signal-filt_signal) 
        corr_values.append(res.correlation) 
        
    th_opt = th_list[np.array(corr_values).argmin()]
    print(th_opt) 
    denoised_signal = fft_denoise(signal, th_opt) 
    return denoised_signal 
    
m0 = 1 
m1 = 0.5 
print('m0', m0, 'm1', m1) 

x = np.linspace(-np.pi/2, np.pi/2, 100) 

cos_func = lambda x: m0 + m1 * np.cos(2*x) 
cos_noise = lambda x: cos_func(x) + np.random.normal(0.0, .5, x.shape[0]) 

# fft = np.fft.rfft(cos_noise(x)) / x.shape[0] 
cos_filter = cos_noise(x)
cos_filter = savgol_filter(cos_noise(x), int( np.ceil(x.shape[0]/10) * 2  + 1), polyorder=0, deriv=0, axis=0, mode='wrap') 

cos_denoise = find_threshold(cos_filter) 

fft = np.fft.fft(cos_denoise) / x.shape[0] 
m0_fft = np.absolute( fft[...,0] ) 
m1_fft = 2 * np.absolute( fft[...,1] ) 

print('m0_fft', m0_fft, 'm1_fft', m1_fft) 

cos_fft = lambda x: m0_fft + m1_fft * np.cos(2*x)

# cos_denoise = fft_denoise(cos_noise(x), .4) 

params, params_covariance = optimize.curve_fit(cos_fit_func, x, cos_noise(x), p0=[1, 1, 1], method='lm') 

m0_fit = params[0]
m1_fit = params[1]
phi_fit = params[2]

cos_fit = m0_fit + m1_fit * np.cos(2*x + phi_fit)

cos_smooth = circular_convolution(cos_noise(x), int(x.shape[0]/10) ) 
m1 = compute_m1(cos_smooth) 

print(m1)

plt.plot(x, cos_noise(x), 'b' ) 
plt.plot(x, cos_fit, 'r' ) 
# plt.plot(x, cos_fft(x), 'r' ) 

