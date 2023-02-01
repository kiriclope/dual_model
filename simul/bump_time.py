import os
from write import *

path = os.getcwd()

replace_global('IF_INI_COND', 0) 
replace_global('INI_COND_ID', 1) 

replace_global('IF_TRIALS', 0) 
replace_global('TRIAL_ID', 2) 

replace_global('FIX_CON', 0) 
replace_global('CON_SEED', 3) 

replace_global('PHI_CUE', 0.25) 
replace_global('PHI_DIST', '1 - PHI_CUE') 
# replace_global('PHI_CUE', 'TRIAL_ID/N_TRIALS') 
# replace_global('PHI_DIST', '0.25 + TRIAL_ID/N_TRIALS') 

# replace_global('KAPPA', 0.25)
replace_global('A_CUE', 0.25) 
replace_global('EPS_CUE', .25) 
replace_global('A_DIST', 0.0) 
replace_global('EPS_DIST', .25) 

# replace_global('folder', "'cos_EI_on'")
replace_global('folder', "'christos_on'")
exec(open( path + '/m1_phi_time.py').read()) 

replace_global('folder', "'christos_off'")
# replace_global('folder', "'cos_EI_off'")
exec(open( path + '/m1_phi_time.py').read()) 

replace_global('IF_INI_COND', 0) 
replace_global('INI_COND_ID', 1) 

replace_global('IF_TRIALS', 0) 
replace_global('TRIAL_ID', 1) 

replace_global('PHI_CUE', 0.25) 
