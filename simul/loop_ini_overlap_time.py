import os

from write import *

path = os.getcwd()

replace_global('IF_INI_COND', 1)
replace_global('IF_TRIALS', 0)

for i_ini in range(11): 
    replace_global('INI_COND_ID', i_ini)
    try:
        exec(open( path + '/overlap_time.py').read())
    except:
        pass
    
replace_global('IF_INI_COND', 0)
