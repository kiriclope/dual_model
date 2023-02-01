import os

from write import *

path = os.getcwd()

replace_global('IF_INI_COND', 1)
replace_global('INI_COND_ID', 1) 
replace_global('IF_TRIALS', 1)

for i_ini in range(1,10+1): 
    replace_global('TRIAL_ID', i_ini) 
    exec(open( path + '/m1_phi_time.py').read()) 

replace_global('IF_TRIALS', 0)
replace_global('IF_INI_COND', 0)
