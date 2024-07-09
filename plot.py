import pandas as pd
import numpy as np
from auxfunctions import Zenith, read_forcing_data
from pucm import PUCM
from auxfunctions import compute_rmse_with_time_shift
from caseutils import periods_to_optimize, periods_qmi, facet_shift, optionsopt, testoptions
import os 
import matplotlib.pyplot as plt

# ====================================================
# Site, model and optimization options
Lat              = 0.7043    # Latitude (positive north)
Lon              = 1.3029    # Longitude (postive west)
offset           = -5        # From UTC to local time
dt               = 10        # timestep in seconds

# ====================================================
CASE        = [ 'artificial0', 'artificial1', 'artificial2', 'artificial3']
mlabels     = [ 'no rain', 'art. rain (runoff+thermal)', 'art. rain (runoff only)', 'art. rain (runoff+thermal+small hf)']
mcolors     = ['blue', 'red', 'green', 'orange']
#CASE        = 'artificial3'
resultsfile = 'ResultsFile/%s.csv'%CASE
site        = 'EquadRoof2010'

# Check if case is a string

if isinstance(CASE, str) and CASE in optionsopt.keys():
    print('Optimizing case: %s'%CASE)
    case_opt         = optionsopt[CASE]['case_opt']
    dbegin           = periods_to_optimize[case_opt][0]
    dfinal           = periods_to_optimize[case_opt][1]
    opt_var          = optionsopt[CASE]['opt_var']
    TurbFlux         = optionsopt[CASE]['TurbFlux']
    runoffmodel      = optionsopt[CASE]['runoffmodel']
    time_shift       = facet_shift[opt_var]
    
    # Reading and preparing data
    df          = read_forcing_data( site, dbegin, dfinal, dt, opt_var)
    Tsecs       = (df.index[-1] - df.index[0]).total_seconds()
    ts          = np.linspace( 0, Tsecs, int(Tsecs/dt))
    nt          = len(ts)

    qz, qs         = Zenith( df, offset=offset, Lat=Lat, Lon=Lon)
    df['qzenith']  = qz
    df['qazimuth'] = qs
    del qz, qs

    Data                = {}
    Data['dt']          = dt
    Data['TurbFlux']    = TurbFlux
    Data['ts']          = ts
    Data['nt']          = nt
    Data['df']          = df
    Data['opt_var']     = opt_var
    Data['runoffmodel'] = runoffmodel
    Data['qmi_start']   = periods_qmi[case_opt]

    if os.path.isfile(resultsfile):
        final = pd.read_csv(resultsfile, index_col=0)
        final.index = pd.to_datetime(final.index)
    else:
        params = {}
        final  = PUCM( Data, params, output='other', verbose=True)
        final.to_csv(resultsfile)
elif isinstance(CASE, str) and CASE in testoptions.keys():
    # Reading and preparing data
    dbegin      = testoptions[CASE]['period'][0]
    dfinal      = testoptions[CASE]['period'][1]
    opt_var     = 'TG1'
    df          = read_forcing_data( site, dbegin, dfinal, dt, opt_var)
    Tsecs       = (df.index[-1] - df.index[0]).total_seconds()
    ts          = np.linspace( 0, Tsecs, int(Tsecs/dt))
    nt          = len(ts)

    qz, qs         = Zenith( df, offset=offset, Lat=Lat, Lon=Lon)
    df['qzenith']  = qz
    df['qazimuth'] = qs
    del qz, qs

    Data                = {}
    Data['dt']          = dt
    Data['TurbFlux']    = testoptions[CASE]['TurbFlux']
    Data['ts']          = ts
    Data['nt']          = nt
    Data['df']          = df
    Data['opt_var']     = opt_var
    Data['runoffmodel'] = testoptions[CASE]['runoffmodel']
    Data['qmi_start']   = testoptions[CASE]['qmi_start']

    if os.path.isfile(resultsfile):
        final = pd.read_csv(resultsfile, index_col=0)
        final.index = pd.to_datetime(final.index)
    else:
        params = {}
        final  = PUCM( Data, params, output='other', verbose=True)
        final.to_csv(resultsfile)

    ##original = '/Users/einaraz/Dropbox/UCMsWRF/pyPUCM/Original/OriginalPUCMequad2010original.csv'
    ##original = pd.read_csv(original, index_col=0)
    ##original.index = pd.to_datetime(original.index)
    ##original = original[final.index[0]:final.index[-1]].copy()
    
    # Create a list of the indices that are present in both df and final and that are not NaN for TG1
    common_indices = df.index.intersection(final.index).tolist()
    OBS =    df.loc[ common_indices ][['TG1','TG2', 'TG3', 'TR1', "swc1", 'swc2', 'swc3','flag']].copy()
    MOD = final.loc[ common_indices ][['TG1','TG2', 'TG3', 'TR1', 'LEG1', 'LEG2', 'LEG3', 'LER', 'WGv1', 'WGv2', 'GrunoffHeight1', 'GrunoffHeight2', 'RrunoffHeight']].copy()
    
    # I'm going to rename all columns in OBS append obs
    OBS.columns = [col + '_obs' for col in OBS.columns]
    OBS.rename(columns={'flag_obs':'flag'}, inplace=True)
    
    # I'm going to concatenate OBS and MOD along the columns
    df       = pd.concat([OBS, MOD], axis=1)
    df_flags = df[df['flag'] == 1].copy()
    
    for Tob,Tme in [ ('TG1_obs', 'TG1'), ('TG2_obs', 'TG2'), ('TG3_obs', 'TG3'), ('TR1_obs', 'TR1'), ("swc3_obs", "WGv1")]:
    #for Tob,Tme in [ ("swc3_obs", "WGv1") ]:
        fig, ax = plt.subplots(1,1)
        rmse_orig, rmse_max = compute_rmse_with_time_shift( df_flags[[Tob,Tme]].copy(), Tob, Tme, 30) #( df_flags[Tob], df_flags[Tme])
        ax.set_title("RMSEorig: %.2f K, RMSEmax: %.2f K"%(rmse_orig, rmse_max))
        if Tob == "swc3_obs":
            for tob in ["swc1_obs", "swc2_obs", "swc3_obs"]:
                ax.scatter( df.index,    df[tob], label=tob)
            for tme in ["WGv1"]:
                ax.plot(    df.index, final[tme],    label=tme)
                #ax.plot(    original.index, original[tme], label=tme + "_orig", linestyle='--')
                #print(original[tme])
        else:
            ax.scatter( df.index,    df[Tob], label='Observed', color='black')
            ax.plot(    df.index, final[Tme], label='modeled' , color='red')
        if Tme == 'TG1':
            ax.twinx().plot( df.index, df['GrunoffHeight1']*1000,  color='orange')
            ax.twinx().plot( df.index, df['LEG1'],  color='blue')
        elif Tme == 'TG2':
            ax.twinx().plot( df.index, df['GrunoffHeight2']*1000,  color='orange')
            ax.twinx().plot( df.index, df['LEG2'],  color='blue')
        elif Tme == 'TR1':
            ax.twinx().plot( df.index, df['RrunoffHeight']*1000,  color='orange')
            ax.twinx().plot( df.index, df['LER'],  color='blue')
    
        ax.grid()
        ax.set_ylabel('Temperature [K]')
        ax.set_xlabel('Time')
        ax.legend()
        plt.show()
        plt.close()
        input("...")
# check if CASE is a list of strings
elif isinstance(CASE, list):
    # define figure with 4 subplots
    fig, axi = plt.subplots(3,1,  figsize=(12, 8))
    ax = [ axi[0], axi[1], axi[2] ]
    fig.subplots_adjust(hspace=0.1, wspace=0.1, left=0.05, right=0.95, top=0.98, bottom=0.1)

    dbegin      = testoptions[CASE[0]]['period'][0]
    dfinal      = testoptions[CASE[0]]['period'][1]
    opt_var     = 'TG1'
    df          = read_forcing_data( site, dbegin, dfinal, dt, opt_var)

    # Plot observed data
    ax[0].scatter(df.index, df['TG1'], color='black', s=1, marker='x')
    ax[1].scatter(df.index, df['TG2'], color='black', s=1, marker='x')
    #ax[3].scatter(df.index, df['TG3'], color='black', s=1, marker='x')
    ax[2].scatter(df.index, df['TR1'], color='black', s=1, marker='x', label='Observed')
    

    for ci,case in enumerate(CASE):
        # read results
        resultsfile = 'ResultsFile/%s.csv'%case
        if os.path.isfile(resultsfile):
            final = pd.read_csv(resultsfile, index_col=0)
            final.index = pd.to_datetime(final.index)
        else:
            print('Case %s not found'%case)
            continue
        ax[0].plot(final.index, final['TG1'], color=mcolors[ci] )
        ax[1].plot(final.index, final['TG2'], color=mcolors[ci] )
        ax[2].plot(final.index, final['TR1'], color=mcolors[ci], label=mlabels[ci])
        # plot runoff heights
        ax[0].twinx().plot(final.index, final['GrunoffHeight1']*1000, linestyle='--', alpha=0.5, color=mcolors[ci])
        ax[1].twinx().plot(final.index, final['GrunoffHeight2']*1000, linestyle='--', alpha=0.5, color=mcolors[ci])
        ax[2].twinx().plot(final.index, final['RrunoffHeight']*1000,  linestyle='--', alpha=0.5, color=mcolors[ci])
    # grid every subplot
    for a in ax:
        a.grid()
    fig.legend(loc='lower center', ncol=12)
    # set x and ylabels
    ax[0].set_ylabel('TG1 [K]')
    ax[1].set_ylabel('TG2 [K]')
    #ax[2].set_ylabel('TG3 [K]')
    ax[2].set_ylabel('TR [K]')
    ax[2].set_xlabel('Time')
    plt.show()
    plt.close()
        
 