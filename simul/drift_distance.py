import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat

from my_statistics import my_bootstraped_ci

import params as gv 
gv.init_param()

pal = [sns.color_palette('tab10')[0],
       sns.color_palette('tab10')[1]]

def classToStim(df):
    
    first_X = [12, 12, 12, 12, 12, -12, -12, -12, -12, -12,
               12, 12, 12, 12, np.nan, -12, -12, -12, -12, np.nan]
    first_Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, np.nan, 0, 0, 0, 0, np.nan]
    
    second_X = [-12, 0, 8.48528137423857, 12, np.nan, 12, 0, -8.48528137423857, -12, np.nan,
                -12, 0, 8.48528137423857, 12, 12, 12, 0, -8.48528137423857, -12, -12]
    second_Y = [0, 12, 8.48528137423857, 0, np.nan, 0, 12, 8.48528137423857, 0, np.nan,
                0, 12, 8.48528137423857, 0, 0, 0, 12, 8.48528137423857, 0, 0]
    

    for i_class in range(20):
        df.loc[df["class"]==i_class+1, "FirstStiX"] = first_X[i_class]
        df.loc[df["class"]==i_class+1, "FirstStiY"] = first_Y[i_class]
        
        df.loc[df["class"]==i_class+1, "SecondStiX"] = second_X[i_class]
        df.loc[df["class"]==i_class+1, "SecondStiY"] = second_Y[i_class]
    
    return df

def carteToPolar(x,y):
    radius = np.sqrt(x*x + y*y)
    theta = np.arctan2(y,x)

    return radius, theta

def get_theta_loc(theta_end, theta_stim, theta_cue, CUT_OFF=[45,135], task='first'):
    
    stim = [0, np.pi]
    # stim = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    cue = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    
    thetas_out = []
    thetas_in = []

    print('get theta at each location')
        
    for i in range(len(stim)):
        idx_stim =  np.where(theta_stim==stim[i])[0].tolist()
        
        for j in range(len(cue)):
            idx_cue = np.where(theta_cue==cue[j])[0].tolist() 
            
            delta = np.abs(stim[i]-cue[j]) * 180/np.pi 
            
            if( delta >= CUT_OFF[0] and delta <= CUT_OFF[1]): 
                idx = list( set(idx_stim) & set(idx_cue) )
                
                print('stim', stim[i]*180/np.pi, 'cue', cue[j]*180/np.pi, 'delta', delta, 'idx', len(idx))
                # print('idx_stim', len(idx_stim), 'idx_cue', len(idx_cue), 'idx', len(idx))                     
                thetas_out.append( theta_end.iloc[idx].to_numpy())
                
                if(task=='first'):
                    thetas_in.append( theta_stim.iloc[idx].to_numpy() )                    
                else:
                    thetas_in.append( theta_cue.iloc[idx].to_numpy() ) 
    
    return np.array(thetas_out), np.array(thetas_in) 


def circular_plot(x, y, figname, color='b'):
    plt.figure(figname, figsize=(3.5, 3.5))
    plt.plot(x, y, 'x', ms=5, color=color)
    plt.axis('off')

    
def correct_sessions(df, task='first'):
    correct_df = df 
    # print('df', df.shape)
    
    correct_df = df[df.State_code==7]
    # print('correct_df', correct_df.shape) 
    
    correct_df = correct_df[correct_df.latency!=0]
    # print('latency', correct_df.shape) 
    if task=='first':
        correct_df = correct_df[correct_df['class']<=10]
    else:
        correct_df = correct_df[correct_df['class']>10]
    
    # print('correct_first_df', correct_first_df.shape) 
    
    return correct_df


def import_session(df, n_session, on=0, monkey=0):

    
    if monkey:
        if on: 
            if n_session>=39 and n_session<=47: 
                filename = 'HECbeh_odrstim%d' % n_session # 39-47                
            elif n_session>=18 and n_session<=29:
                filename = 'HECstim0%d_1' % n_session # 18-29
        
        else:
            if n_session<=15 and n_session>=2:
                filename = 'HECbeh_odrstim_%d' % n_session # 2-15
            elif n_session>=25 and n_session<=45 :
                filename = 'HECbeh_odr%d' % n_session # 25 to 45            
    else:
        if on: 
            filename = "'GRUstim0%d_1'" % n_session # 28-44 
        else:
            if n_session<=10:
                filename = "'GRUbeh_odrstim_%d'" % n_session # 1-10
            else:
                filename = "'GRUbeh_odr%d'" % (n_session-2) # 9-17
    
    print('filename', filename, np.sum(df.filename == filename))
    
    if np.sum(df.filename == filename)>0 :
        return df[df.filename == filename] 
    else:
        return np.nan
    
def get_X_Y(df):

    X_end = df.endpointX
    Y_end = df.endpointY
    
    X_stim = df.FirstStiX
    Y_stim = df.FirstStiY

    X_cue = df.SecondStiX
    Y_cue = df.SecondStiY

    return X_end, Y_end, X_stim, Y_stim, X_cue, Y_cue

def get_phases(df):

    X_end = df.endpointX
    Y_end = df.endpointY
    
    X_stim = df.FirstStiX
    Y_stim = df.FirstStiY

    X_cue = df.SecondStiX
    Y_cue = df.SecondStiY
    
    _, theta_end = carteToPolar(X_end, Y_end)
    # print('theta', theta_end.shape)
    
    _, theta_stim = carteToPolar(X_stim, Y_stim)
    # print('theta_stim', theta_end.shape)

    _, theta_cue = carteToPolar(X_cue, Y_cue)
    
    return theta_end, theta_stim, theta_cue


def get_drift(theta_out, theta_in, THRESH=30):
    
    drift = (theta_out - theta_in) 
    
    drift[drift>np.pi] -= 2*np.pi 
    drift[drift<-np.pi] += 2*np.pi 
    drift *= 180/np.pi 
    
    drift = drift[np.abs(drift)<THRESH] 
    drift = np.abs(drift)
    
    print('DRIFT: stim/cue', np.nanmean(theta_in)*180/np.pi, 'mean drift', np.nanmean(drift))
    
    return drift


def get_diff(theta_out, theta_in, THRESH=30):

    # average over trials
    mean_theta = stat.circmean(theta_out, nan_policy='omit', axis=0, high=np.pi, low=-np.pi)        
    diff = (theta_out - mean_theta) 

    # print(diff* 180/np.pi )
    
    # diff[diff>np.pi] -= 2*np.pi  
    # diff[diff<-np.pi] += 2*np.pi 
    diff *= 180/np.pi 
    
    print('DIFF: stim/cue', np.nanmean(theta_in)*180/np.pi, 'mean_theta', mean_theta*180/np.pi, 'mean diff', np.nanmean(diff)) 
    
    diff = diff[np.abs(diff)<THRESH] 
    # diff = np.abs(diff)
    return diff

if __name__ == "__main__":
    
    IF_SESSION = 0
    task = 'sec'
    monkey = 2
    THRESH = 90
    CUT_OFF = [0, 180] 
    cut_off = [0, 45, 90, 180]
    
    if monkey==2: 
        df = pd.read_excel('./data/StimulationEyeEndPoint2.xlsx', engine='openpyxl', sheet_name='Inf_HecNoStim')
        df = classToStim(df)
        df2 = pd.read_excel('./data/StimulationEyeEndPoint2.xlsx', engine='openpyxl', sheet_name='Inf_GruNoStim')
        df = df.append(df2)
    elif monkey==1:
        df = pd.read_excel('./data/StimulationEyeEndPoint2.xlsx', engine='openpyxl', sheet_name='Inf_HecNoStim')
        df = classToStim(df)
    else:
        df = pd.read_excel('./data/StimulationEyeEndPoint2.xlsx', engine='openpyxl', sheet_name='Inf_GruNoStim')
    
    df['filename'] = df['filename'].astype("string")
    df_correct = correct_sessions(df, task=task)     
    theta_end, theta_stim, theta_cue = get_phases(df_correct)    
    
    drift_D_off = [] 
    diff_D_off = []
    
    ci_drift_off = []
    ci_diff_off = []
    
    for i in range(4):
        drift_off = []
        diff_off = []
        
        CUT_OFF = [cut_off[i], cut_off[i]] 
        
        thetas_out, thetas_in = get_theta_loc(theta_end, theta_stim, theta_cue, CUT_OFF, task=task) 
        
        for i_stim in range(thetas_in.shape[0]):
            drift_off.append( get_drift(thetas_out[i_stim], thetas_in[i_stim], THRESH))            
            diff_off.append( get_diff(thetas_out[i_stim], thetas_in[i_stim], THRESH) ) 
            
        drift_off = np.hstack(drift_off)
        diff_off = np.hstack(diff_off)

        mean_off, ci_off = my_bootstraped_ci(drift_off, n_samples=10000) 

        _, ci_off_2 = my_bootstraped_ci(np.abs(diff_off), n_samples=10000) 
        
        print('distance', cut_off[i], 'drift_off', drift_off.shape, 'diff_off', diff_off.shape) 
        print('mean', np.nanmean(drift_off), 'boots mean', mean_off, 'ci_off', ci_off) 
        
        mean_drift = np.nanmean( np.abs(drift_off) )
        # var_diff = np.nanvar(diff_off) 
        var_diff = np.nanmean( np.abs(diff_off) )
        
        drift_D_off.append(mean_drift)
        diff_D_off.append(var_diff)

        ci_drift_off.append(ci_off)
        ci_diff_off.append(ci_off_2)
        
    # ######################
    # # on condition
    # ######################
    
    if monkey==2:
        df = pd.read_excel('./data/StimulationEyeEndPoint2.xlsx', engine='openpyxl', sheet_name='Inf_HecStim')
        df = classToStim(df)         
        df2 = pd.read_excel('./data/StimulationEyeEndPoint2.xlsx', engine='openpyxl', sheet_name='Inf_GruStim')
        df = df.append(df2)
    elif monkey==1:
        df = pd.read_excel('./data/StimulationEyeEndPoint2.xlsx', engine='openpyxl', sheet_name='Inf_HecStim')
        df = classToStim(df)         
    else:
        df = pd.read_excel('./data/StimulationEyeEndPoint2.xlsx', engine='openpyxl', sheet_name='Inf_GruStim')
    
    df['filename'] = df['filename'].astype("string")     
    df_correct = correct_sessions(df, task=task) 
    theta_end, theta_stim, theta_cue = get_phases(df_correct)    
    
    drift_D_on = [] 
    diff_D_on = []
    
    ci_drift_on = []
    ci_diff_on = []
    
    for i in range(4):
        drift_on = []
        diff_on = []
        
        CUT_OFF = [cut_off[i], cut_off[i]] 
        
        thetas_out, thetas_in = get_theta_loc(theta_end, theta_stim, theta_cue, CUT_OFF, task=task) 
        
        for i_stim in range(thetas_in.shape[0]):
            drift_on.append( get_drift(thetas_out[i_stim], thetas_in[i_stim], THRESH)) 
            diff_on.append( get_diff(thetas_out[i_stim], thetas_in[i_stim], THRESH) ) 
        
        drift_on = np.hstack(drift_on)
        diff_on = np.hstack(diff_on)

        mean_on, ci_on = my_bootstraped_ci(drift_on, n_samples=10000) 
        _, ci_on_2 = my_bootstraped_ci(np.abs(diff_on), n_samples=10000) 
        
        print('distance', cut_off[i], 'drift_on', drift_on.shape, 'diff_on', diff_on.shape) 
        print('mean', np.nanmean(drift_on), 'boots mean', mean_on, 'ci_on', ci_on)         
        
        mean_drift = np.nanmean( np.abs(drift_on) )
        var_diff = np.nanvar(diff_on)
        var_diff = np.nanmean(np.abs(diff_on) )
        
        drift_D_on.append(mean_drift)
        diff_D_on.append(var_diff)
    
        ci_drift_on.append(ci_on)
        ci_diff_on.append(ci_on_2)
    
    plt.figure('drift_distance', figsize=(5.663, 3.5))
    plt.plot(cut_off, drift_D_off, '-o', color=pal[0])
    plt.plot(cut_off, drift_D_on, '-o', color=pal[1])
    plt.xticks([0, 45, 90, 180])
    plt.xlabel('Distance (°) ')
    plt.ylabel('Drift (°) ')
    for i in range(4):
        plt.errorbar(cut_off[i], ci_drift_off[i], color=pal[0]) 
        plt.errorbar(cut_off[i], ci_drift_on[i], color=pal[1]) 
        
    plt.figure('diff_distance', figsize=(5.663, 3.5))
    plt.plot(cut_off, diff_D_off, '-o', color=pal[0])
    plt.plot(cut_off, diff_D_on, '-o', color=pal[1])
    plt.xticks([0, 45, 90, 180])
    plt.xlabel('Distance (°) ')
    plt.ylabel('Diffusion (°) ')
    for i in range(4):
        plt.errorbar(cut_off[i], ci_diff_off[i], color=pal[0]) 
        plt.errorbar(cut_off[i], ci_diff_on[i], color=pal[1]) 
    
