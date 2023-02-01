import numpy as np

global n_pop, n_neurons, K, m0, folder 
n_pop = 2
n_neurons = 2 
K = 500 
folder = 'L23' 

global ext_inputs, J, Tsyn 
ext_inputs = []
J = []
Tsyn = []

global J0, I0
if n_pop==1:
    J0 = 1.0
    I0 = 0.1

filename = 'inputs.dat'

global IF_TRIALS, TRIAL_ID
IF_TRIALS = 0
TRIAL_ID = 1 

global IF_LOW_RANK, MEAN_XI, VAR_XI
K = 2 
MEAN_XI = -0.0
VAR_XI = 5.0 

global IF_LEFT_RIGHT, MEAN_XI_LEFT, MEAN_XI_RIGHT, VAR_XI_LEFT, VAR_XI_RIGHT, RHO
IF_LEFT_RIGHT = 0
MEAN_XI_LEFT = -0.0
VAR_XI_LEFT = 5.0 
MEAN_XI_RIGHT = -0.0
VAR_XI_RIGHT = 5.0 
RHO = 0.5

global IF_FF, MEAN_FF, VAR_FF, VAR_ORTHO, IF_RHO_FF, RH0_FF_XI
IF_FF = 0
MEAN_FF = 1.0
VAR_FF = 1.0
VAR_ORTHO = 0.0

IF_RHO_FF = 0
RHO_FF_XI = 1.0 
