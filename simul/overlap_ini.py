import os

from write import *

path = os.getcwd()

replace_global('IF_INI_COND', 1)

replace_global('alpha', .2)

for i_ini in range(1, 10+1): 
    replace_global('INI_COND_ID', i_ini)
    if(i_ini==1):
        replace_global('IF_ADD_VLINES', 1)
    else:
        replace_global('IF_ADD_VLINES', 0)
    
    exec(open( path + '/overlap_time.py').read()) 

replace_global('IF_INI_COND', 0) 
replace_global('INI_COND_ID', 1) 
replace_global('IF_ADD_VLINES', 1)
replace_global('alpha', 1)
