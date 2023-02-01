import os

from write import *

path = os.getcwd()

replace_global('IF_INI_COND', 1)
replace_global('IF_TRIALS', 1)


for i_ini in range(1, 4+1): 
    replace_global('INI_COND_ID', i_ini) 
    for i_trial in range(4,4+1): 
        replace_global('TRIAL_ID', i_trial) 
        try: 
            exec(open( path + '/m1_phi_time.py').read()) 
        except: 
            pass 
        
replace_global('IF_INI_COND', 0)
replace_global('IF_TRIALS', 0)
   
