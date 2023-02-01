import os

from write import *

path = os.getcwd()

for i_ini in range(10):
    replace_global('INI_COND_ID', i_ini)
    exec(open( path + '/animated_bump.py').read()) 
