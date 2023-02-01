import numpy as np

global n_pop, n_neurons, K, m0, folder 
n_pop = 2
n_neurons = 2

n_size = int(n_neurons / n_pop * 10000)

# if n_size==0:
#     n_size = 1
    
K = 2000 
folder = 'L23' 
m0 = .1 

global ext_inputs, J, Tsyn 
ext_inputs = [] 
J = [] 
Tsyn = [] 

global pal
pal = ['r', 'b'] 

global J0, I0 
J0 = 1.0
I0 = 0.5

if n_pop==1 :
    folder = 'I0_%.2f_J0_%.2f' % (I0, J0) 

global filename
filename = 'inputs.dat'

global IF_STP 
IF_STP = 0 

global IF_TRIALS, TRIAL_ID
IF_TRIALS = 0 
TRIAL_ID = 1 

global IF_DPA, IF_DUAL
IF_DPA = 0  
IF_DUAL = 0 

global IF_STRUCTURE, IF_SPEC, IF_RANK_2
IF_SPEC = 1
IF_RANK_2 = 0 
IF_STRUCTURE = IF_SPEC or IF_RANK_2

global KAPPA, KAPPA_1 
KAPPA = 30
KAPPA_1 = 30 

global IF_LOW_RANK, MEAN_XI, VAR_XI
IF_LOW_RANK = 0 
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

IF_RHO_FF = 1
RHO_FF_XI = 1.0 

def init_param():
    global folder, n_pop, n_neurons, K, ext_inputs, J, m0, Tsyn, path, KAPPA, KAPPA_1 
    
    path = '' 
    
    if(n_pop!=1): 
        print("reading parameters from:")
        file_name = "/homecentral/alexandre.mahrach/IDIBAPS/cpp/model/parameters/%dpop/%s.txt" % (n_pop, folder) 
        print(file_name)
        
        i=0
        with open(file_name, 'r') as file:  # Open file for read
            for line in file:           # Read line-by-line
                line = line.strip().split()  # Strip the leading/trailing whitespaces and newline
                line.pop(0)
                if i==0:
                    ext_inputs = np.asarray([float(j) for j in line])
                    ext_inputs *= m0
                    # print(ext_inputs)
                if i==1:
                    J = np.asarray([float(j) for j in line])
                    J = J.reshape(n_pop, n_pop)
                    # print(J)
                if i==2:
                    Tsyn = np.asarray([float(j) for j in line])
                    Tsyn = Tsyn.reshape(n_pop, n_pop)
                    # print(Tsyn)
                i=i+1
    else:
        ext_inputs = I0
        J = J0
        Tsyn = 2 ;
        
    path = '/homecentral/alexandre.mahrach/IDIBAPS/cpp/model/simulations/%dpop/%s/N%d/K%d/' % (n_pop, folder, n_neurons, K)
    
    if(IF_STRUCTURE):
        if(IF_SPEC):
            path += 'spec/kappa_%.2f/' % KAPPA 
        elif(IF_RANK_2):
            path += 'spec/kappa_%.2f_kappa_1_%.2f/' % (KAPPA, KAPPA_1)
            
    if(IF_DPA) : 
        path += 'DPA/'
    elif(IF_DUAL) :
        path += 'dual_task/'
            
    # path = '/homecentral/alexandre.mahrach/IDIBAPS/cpp/model/simulations/2pop/L23/N5/K5000/spec/kappa_16_kappa_1_16/'
    
    if(IF_LOW_RANK):
        if(IF_LEFT_RIGHT):
            path += 'low_rank/xi_left_mean_%.2f_var_%.2f' % (MEAN_XI_LEFT,VAR_XI_LEFT)
            path += '_xi_right_mean_%.2f_var_%.2f/' % (MEAN_XI_RIGHT,VAR_XI_RIGHT)
            path += 'rho_%.2f/' % RHO
        else:
            path += 'low_rank/xi_mean_%.2f_var_%.2f/' % (MEAN_XI,VAR_XI)
        if(IF_FF):
            if(IF_RHO_FF):
                path += 'ff_mean_%.2f_var_%.2f_rho_%.2f/' % (MEAN_FF,VAR_FF,RHO_FF_XI)
            else:
                path += 'ff_mean_%.2f_var_%.2f_ortho_%.2f/' % (MEAN_FF,VAR_FF,VAR_ORTHO)
                
    if(IF_TRIALS):
        path += 'trial_%d/' % TRIAL_ID ;

    print('reading simulations data from:')
    print(path)
    
