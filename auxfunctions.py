import numpy as np
import math 
from scipy import special
import pandas as pd
from datetime import datetime, timedelta
from constants import Constants, pucmparams

def read_forcing_data(case, dbegin, dfinal, dt, opt_var):
    """
    Read forcing data necessary for simulation
    dbegin: initial date of simulation
    dfinal: final date of simulation
    dt: time step in seconds for interpolation of data (same time step used in model)
    """
    # Change how precipitation is interpolated
    
    if case == 'EquadRoof2010':
        # Read forcing data from EquadRoof2010
        # qa [kg/kg]; Pa [Pa]; Ld, Sd [Wm^2]; Ta [C]; ra [kg/m3]; Ua [m/s]; Pd [m/s]
        df       = pd.read_csv('/Users/einaraz/Dropbox/UCMsWRF/pyPUCM/InputData/EquadRoof2010RegressionFilled.csv', index_col=0)
        df.index = pd.to_datetime(df.index)
        df       = df[(df.index>=dbegin)&(df.index<=dfinal)].copy()
        df       = df.resample('%ss'%dt).asfreq().interpolate(method='linear')
        
        df['Ta']               = df['Ta'] + 273.15  # virtual air temperature [oC]
        #df['Ua'][df['Ua']<0.1] = 0.1     
        df.loc[df['Ua'] < 0.1, 'Ua'] = 0.1

        TsurfObs         = pd.read_csv("/Users/einaraz/Dropbox/UCMsWRF/DataEquadRoof2010/SNOPsensorNetwork/Equad2010SurfaceTemperature.csv", index_col=0)
        TsurfObs.index   = pd.to_datetime(TsurfObs.index)
        TsurfObs         = TsurfObs[(TsurfObs.index>=dbegin)&(TsurfObs.index<=dfinal)].copy()
        TsurfObs         = TsurfObs[~TsurfObs.index.duplicated(keep='first')].copy()
        TsurfObs         = TsurfObs.resample('1min').first()
        TsurfObs['swc1'] = TsurfObs['swc1']/100
        TsurfObs['swc2'] = TsurfObs['swc2']/100
        TsurfObs['swc3'] = TsurfObs['swc3']/100

        TsurfObs.rename(columns={'Troof': "TR1", 
                              "Tasphalt": "TG1",
                             "Tconcrete": "TG2",
                               "Tgrass1": "TG3",
                                  "swc1": 'swc1',
                                  "swc2": 'swc2',
                                  "swc3": 'swc3',
                                  },  inplace=True)
        
        #TsurfObs         = TsurfObs.resample('%sS'%dt).asfreq().interpolate(method='linear', limit=10)
        df = pd.concat([df, TsurfObs], axis=1)
        df.index = df.index - timedelta(hours=5)
        # Define target value and flags
        df['Obs']   = df[opt_var].copy()
        df['flag']  = np.ones(df.index.size)
        # set flag to 0 when df['TG1'] is NaN
        df.loc[df['Obs'].isnull(), 'flag'] = 0
        # In addition, make sure the first day the flag is also zero
        all_days = sorted(list(set(list(df.index.date))))
        df.loc[df.index.date <= all_days[1], 'flag'] = 0
        # Ignore a few hours of the day (shading hours)
        if opt_var in ['TG2']: 
            df.loc[(df.index.hour>=6)&(df.index.hour<=12), 'flag']  = 0
        if opt_var in ['TG3']:
            df.loc[(df.index.hour>=12)&(df.index.hour<=17), 'flag']  = 0
            df.loc[(df.index.hour>=6)&(df.index.hour<=7), 'flag']  = 0
        df.loc[(df.index>=datetime(2010,7,12,17))&(df.index<=datetime(2010,7,16)), 'flag']  = 1
        df.index = df.index + timedelta(hours=5)
        
        return df
    elif case == 'EquadRoof2010artificial':
        # Read forcing data from EquadRoof2010
        # qa [kg/kg]; Pa [Pa]; Ld, Sd [Wm^2]; Ta [C]; ra [kg/m3]; Ua [m/s]; Pd [m/s]
        df       = pd.read_csv('/Users/einaraz/Dropbox/UCMsWRF/pyPUCM/InputData/EquadRoof2010RegressionFilledArtificialRain.csv', index_col=0)
        df.index = pd.to_datetime(df.index)
        df       = df[(df.index>=dbegin)&(df.index<=dfinal)].copy()
        df       = df.resample('%ss'%dt).asfreq().interpolate(method='linear')
        
        df['Ta']               = df['Ta'] + 273.15  # virtual air temperature [oC]
        #df['Ua'][df['Ua']<0.1] = 0.1     
        df.loc[df['Ua'] < 0.1, 'Ua'] = 0.1

        TsurfObs         = pd.read_csv("/Users/einaraz/Dropbox/UCMsWRF/DataEquadRoof2010/SNOPsensorNetwork/Equad2010SurfaceTemperature.csv", index_col=0)
        TsurfObs.index   = pd.to_datetime(TsurfObs.index)
        TsurfObs         = TsurfObs[(TsurfObs.index>=dbegin)&(TsurfObs.index<=dfinal)].copy()
        TsurfObs         = TsurfObs[~TsurfObs.index.duplicated(keep='first')].copy()
        TsurfObs         = TsurfObs.resample('1min').first()
        TsurfObs['swc1'] = TsurfObs['swc1']/100
        TsurfObs['swc2'] = TsurfObs['swc2']/100
        TsurfObs['swc3'] = TsurfObs['swc3']/100

        TsurfObs.rename(columns={'Troof': "TR1", 
                              "Tasphalt": "TG1",
                             "Tconcrete": "TG2",
                               "Tgrass1": "TG3",
                                  "swc1": 'swc1',
                                  "swc2": 'swc2',
                                  "swc3": 'swc3',
                                  },  inplace=True)
        
        #TsurfObs         = TsurfObs.resample('%sS'%dt).asfreq().interpolate(method='linear', limit=10)
        df = pd.concat([df, TsurfObs], axis=1)
        df.index = df.index - timedelta(hours=5)
        # Define target value and flags
        df['Obs']   = df[opt_var].copy()
        df['flag']  = np.ones(df.index.size)
        # set flag to 0 when df['TG1'] is NaN
        df.loc[df['Obs'].isnull(), 'flag'] = 0
        # In addition, make sure the first day the flag is also zero
        all_days = sorted(list(set(list(df.index.date))))
        df.loc[df.index.date <= all_days[1], 'flag'] = 0
        # Ignore a few hours of the day (shading hours)
        if opt_var in ['TG2']: 
            df.loc[(df.index.hour>=6)&(df.index.hour<=12), 'flag']  = 0
        if opt_var in ['TG3']:
            df.loc[(df.index.hour>=12)&(df.index.hour<=17), 'flag']  = 0
            df.loc[(df.index.hour>=6)&(df.index.hour<=7), 'flag']  = 0
        df.loc[(df.index>=datetime(2010,7,12,17))&(df.index<=datetime(2010,7,16)), 'flag']  = 1
        df.index = df.index + timedelta(hours=5)
        
        return df

    else:
        raise ValueError('Case not found. No data available for simulation.')

def TwetbulbTdew(T,RH, Constants):
    # Computes wet bulb temperature using Stull's formula
    # T in C and RH in %
    
    # Wet bulb temperature
    Twet = T * np.arctan(0.151977 * (RH + 8.313659)**0.5 ) + \
         np.arctan(T + RH) - np.arctan(RH - 1.676331) + \
         0.00391838 * ( (RH)**(3/2) ) * np.arctan(0.023101 * RH) - 4.686035
         
    # Dew point temperature
    Tdew  = T - (100 - RH)/5.
    return Twet + Constants.KK, Tdew + Constants.KK

def LE_leaf(Rn, Rb, Rs, ra, Ta, qa, qsat):
    #------------------------------------------------------------------------
    # Reference, Green (1993) & Dingman book
    # Rn : net radiation (W m-2)
    # Rb : leaf boundary layer resistance (s m-1)
    # Rs : leaf stomatal resistance (s m-1)
    # ra : air density (kg m-3)
    # Ta : air temperature in K
    # qa : specific humidity, kg/kg
    # qsat : saturated specific humidity, kg/kg
    #------------------------------------------------------------------------
    
    TC    = Ta - 273.15
    s     = 2508.3 * 1.e3/(TC + 237.3)/(TC + 237.3) * np.exp(( 17.3 * TC)/(TC + 237.3))
    gamma = 66.1  # Pa
    Cp    = 1005.
    
    # saturation vapor pressure (Pa), Physical Hydrology, Dingman, p. 273
    esat = 611.0 * np.exp( (17.3*TC) / (TC + 237.3) )
    RH   = qa/qsat
    ea   = esat*RH
    D    = esat - ea   # vapor pressure deficit of air (Pa)
    LE   = ( s * Rn + 0.93 * ra * Cp * D / Rb )/( s + 0.93 * gamma *( 2.0 + Rs/Rb) )
    return LE

def StomatalResistanceTree(Rsmin, Sd, SMC, Wwlt, Wsat, Ta, qa, qsat, rootden, LAItree):
    #------------------------------------------------------------------------
    # Reference, Niyogi and Raman (1997)
    # Sd: solar radiation flux (W m-2)
    # LAI: Leaf Area Index
    # W2 : deep soil moisture (volumetric water content at 1 m below the surface)
    # Wwlt : wilting soil moisture
    # Wsat : saturated soil moisture
    # Ta : air temperature in K
    # qa : specific humidity, kg/kg
    # qsat : saturated specific humidity, kg/kg
    # Pa : atmospheric pressure, Pa
    #------------------------------------------------------------------------
    
    Rnl   = 100      # 100 W m-2, radiation limit at which photosynthesis is assumed to start
    Rsmax = 5000     # 5000 s m-1, maximum stomatal resistance
    # dgG = 2.0/10 * np.ones(shape=(10,1))
    
    f  = 0.55*2.*Sd/Rnl/LAItree
    F1 = (f + Rsmin/Rsmax)/(1.0 + f)    # FSR
    F1 = max(F1, 0.0001)                # following Noah LSM
    W2 = sum(SMC * rootden)
    
    if ( W2 > 0.75 * Wsat ):
        F2 = 1.0
    elif ( (W2 >= Wwlt) & (W2 <= 0.75*Wsat) ):
        F2 = (W2 - Wwlt)/(0.75 * Wsat - Wwlt)
    elif ( W2 < Wwlt ):
        F2 = 0.0
    else: pass

    F2 = max(F2, 0.0001)  # Ftheta
    
    # as in Noah LSM
    HS = 54.53                        # broad-leaf tree
    F3 = 1.0/(1.0 + HS * (qsat - qa)) # Fe
    F3 = max(F3, 0.01)                # Fe following Noah LSM
    
    F4 = 1.0 - 0.0016 * (298.0 - Ta)**2  # FT
    F4 = max(F4, 0.0001)                 # FT - following Noah LSM
    
    Rs = Rsmin/LAItree/F1/F2/F3/F4
    
    return Rs

def LeafBLMresistance(u, Aleaf, aleaf):
    #------------------------------------------------------------------------
    # Reference, Green (1993) & Landsberg and Powell (1973)
    # p: ratio of total leaf plan area to the area of the foliage projected
    # onto a vertical plane
    # a: radius of tree crown
    # u: mean wind speed (m s-1) accross the leaf surface (taken at a mid
    # canopy height)
    # u is assumed to be the same as the canyon wind speed (Young-Hee).
    # d: characteristic leaf dimension, hard-wired with 0.1 m
    #------------------------------------------------------------------------
    p  = Aleaf/(2.0 * aleaf)                # assumed by Young-Hee
    d  = 0.1
    Rb = 58.0 * (p**0.56) * (d/u)**0.5
    return Rb

def StomatalResistance(Rsmin, Sd, LAI, W2, Wwlt, Wsat, Ta, qa, qsat, Pa):
    #------------------------------------------------------------------------
    # Reference, Niyogi and Raman (1997)
    # Rsmin: minimum stomatal resistance
    # Sd: solar radiation flux (W m-2)
    # LAI: Leaf Area Index
    # W2 : deep soil moisture (volumetric water content at 1 m below the surface)
    # Wwlt : wilting soil moisture
    # Wsat : saturated soil moisture
    # Ta : air temperature in K
    # qa : specific humidity, kg/kg
    # qsat : saturated specific humidity, kg/kg
    # Pa : atmospheric pressure, Pa
    #------------------------------------------------------------------------

    Rnl   = 100   # 100 W m-2, radiation limit at which photosynthesis is assumed to start
    Rsmax = 5000  # 5000 s m-1, maximum stomatal resistance
    
    f  = 0.55 * 2 * Sd / Rnl / LAI
    F1 = (f + Rsmin / Rsmax ) / (1. + f)
    F1 = max( F1, 0.0001)    # following Noah LSM
    
    if ( W2 > 0.75 * Wsat ):
        F2 = 1.0
    elif ( (W2 >= Wwlt) & (W2 <= 0.75*Wsat) ):
        F2 = (W2 - Wwlt)/(0.75*Wsat - Wwlt)
    elif ( W2 < Wwlt ):
        F2 = 0.0
    else: pass

    F2 = max(F2, 0.0001)
    
    Pa_hPa = Pa/100.0
    TaC    = Ta - 273.15
    
    RH   = qa/qsat
    esat = 6.112 * np.exp((17.67 * TaC)/(TaC + 243.5))
    ea   = esat * RH
        
    # as in Noah LSM
    HS = 54.53                              # broad-leaf tree
    F3 = 1.0/(1.0 + HS*(qsat - qa))
    F3 = max(F3, 0.01)                      # following Noah LSM
    
    F4 = 1.0 - 0.0016 * (298.0 - Ta)**2
    F4 = max(F4, 0.0001)                    # following Noah LSM
    
    Rs = Rsmin/LAI/F1/F2/F3/F4

    return Rs

def TurbulentFluxesAero2( Ta, Qa, Te, Ts, Qs, U, zref, z0m, z0h):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Raerod_yh.m - June, 19 2013                              %
    # author: Young-Hee Ryu                                    %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #------------------------------------------------------------------------
    #  Purpose:
    #     compute u*, t*, q* to calculate sensible/latent heat flux
    #     Based on Kot and Song (1998, BLM)
    #
    #  Variable Description:
    #     U   - wind speed
    #     Ta  - potential temperature in K of atmosphere
    #     Te  - effective (averaged) temperature in K at surface
    #     Ts  - temperature in K at surface
    #     q   - water content of atmosphere/surface
    #     z0m - aerodynamic roughness for momentum
    #     z0h - aerodynamic roughness for heat
    #------------------------------------------------------------------------

    g  = 9.81           # gravity constant
    k  = 0.4            # Von-Karman constants
    bm = 8.0
    bh = 23.0
    R  = 1.0
    F2 = (1.0 - z0m / zref)**2.0 / (1.0 - z0h/zref)
    RiB = g * zref * (Ta- Te) * F2/ ( (Ta + Te) * 0.5 * U**2) # when stability is determined, effective (averaged) temperature is used.
    Am  = k/np.log( zref /z0m )
    Ah  = k/np.sqrt(np.log(zref/z0m)*np.log(zref/z0h))

    if ( RiB < 0. ):       #unstable condition
        C1     = -0.9848
        C2     = 2.5398
        C3     = -0.2325
        C4     = 14.1727
        Cmstar = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
        
        if ( (z0m/z0h >= 1.0) & (z0m/z0h <= 100.) ):
            C1 = -1.1790
            C2 = -1.9256
            C3 = 0.1007
            C4 = 16.6796
        elif ( (z0m/z0h > 100.) & (z0m/z0h <= 10000.) ):
            C1 = -1.0487
            C2 = -1.0689
            C3 = 0.0952
            C4 = 11.7828
        else: pass

        Chstar = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
        
        cm = Cmstar * (Am**2) * bm * np.sqrt(F2) * ( (zref/z0m)**(1./3.) - 1. )**(3./2.)
        ch = Chstar * (Ah**2) * bh * np.sqrt(F2) * ( (zref/z0h)**(1./3.) - 1. )**(3./2.)
        
        Fm = 1.0 - (bm*RiB)/(1.0 + cm * np.sqrt(abs(RiB)))
        Fh = 1.0 - (bh*RiB)/(1.0 + ch * np.sqrt(abs(RiB)))
        
    else: # stable condition
        if ( (z0m/z0h >= 1.) & (z0m/z0h <= 100.) ):
            C1 = -0.4738
            C2 = -0.3268
            C3 = 0.0204
            C4 = 10.0715
        elif ( (z0m/z0h > 100.) & (z0m/z0h <= 10000.) ):
            C1 = -0.4613
            C2 = -0.2402
            C3 = 0.0146
            C4 = 8.9172
        else: pass

        dm = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
        
        if ( (z0m/z0h >= 1.0) & (z0m/z0h <= 100.) ):
            C1 = -0.5128
            C2 = -0.9448
            C3 = 0.0643
            C4 = 10.8925
        elif ( (z0m/z0h > 100.) & (z0m/z0h <= 10000.) ):
            C1 = -0.3169
            C2 = -0.3803
            C3 = 0.0205
            C4 = 7.5213
        else: pass
        
        dh = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
        
        Fm = 1./(1.0 + dm * RiB)**2
        Fh = 1./(1.0 + dh * RiB)**2
        
    ustar = np.sqrt((Am**2)*(U**2)*Fm)
    tstar = 1.0/R*Ah**2 * U *( Ta - Ts ) * Fh/ustar
    qstar = 1.0/R*Ah**2 * U *( Qa - Qs ) * Fh/ustar
    
    return ustar, tstar, qstar

def TurbulentFluxesAero( Ta, Qa, rho_air, Ts, Qs, U, zref, z0m, z0h, Constants):
    #------------------------------------------------------------------------
    #  Purpose:
    #     compute u*, t*, q* to calculate sensible/latent heat flux
    #     Based on Kot and Song (1998, BLM)
    #
    #  Variable Description:
    #     U   - wind speed
    #     Ta  - potential temperature in K of atmosphere
    #     Ts  - temperature in K at surface
    #     q   - water content of atmosphere/surface
    #     z0m - aerodynamic roughness for momentum
    #     z0h - aerodynamic roughness for heat
    #------------------------------------------------------------------------
    g  = 9.81           # gravity constant
    k  = 0.4            # Von-Karman constants
    bm = 8.0
    bh = 23.0
    R  = 1.0
    F2 = (1.0 - z0m / zref )**2/(1.0 - z0h / zref)
    RiB = g * zref * ( Ta - Ts ) * F2 / ( (Ta + Ts)*0.5*U**2)

    Am  = k/np.log(zref/z0m)
    Ah  = k/np.sqrt(np.log(zref/z0m)*np.log(zref/z0h))

    if ( RiB < 0. ): # unstable condition
        C1 = -0.9848
        C2 = 2.5398 
        C3 = -0.2325
        C4 = 14.1727
        Cmstar = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
    
        if ( (z0m/z0h >= 1.0) & (z0m/z0h <= 100.) ):
            C1 = -1.1790
            C2 = -1.9256
            C3 =  0.1007
            C4 = 16.6796
        elif ( (z0m/z0h > 100.) & (z0m/z0h <= 10000.) ):
            C1 = -1.0487
            C2 = -1.0689
            C3 =  0.0952
            C4 = 11.7828
    
        Chstar = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
    
        cm = Cmstar * Am**2 * bm * np.sqrt(F2) *( (zref/z0m)**(1./3.) - 1.0 )**(3./2.)
        ch = Chstar * Ah**2 * bh * np.sqrt(F2) *( (zref/z0h)**(1./3.) - 1.0 )**(3./2.)
        Fm = 1.0 - (bm*RiB)/(1. + cm * np.sqrt( abs(RiB)) )
        Fh = 1.0 - (bh*RiB)/(1. + ch * np.sqrt( abs(RiB)) )
    
    else:
        if ( (z0m/z0h >= 1.0) & (z0m/z0h <= 100.) ):
            C1 = -0.4738
            C2 = -0.3268
            C3 =  0.0204
            C4 = 10.0715
        elif ( (z0m/z0h > 100.) & (z0m/z0h <= 10000.) ):
            C1 = -0.4613
            C2 = -0.2402
            C3 =  0.0146
            C4 =  8.9172
        else: pass

        dm = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
    
        if ( (z0m/z0h >= 1.0) & (z0m/z0h <= 100.) ):
            C1 =  -0.5128
            C2 =  -0.9448
            C3 =   0.0643
            C4 =  10.8925
        elif ( (z0m/z0h > 100.) & (z0m/z0h <= 10000.) ):
            C1 = -0.3169
            C2 = -0.3803
            C3 =  0.0205
            C4 =  7.5213
        else: pass

        dh = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
    
        Fm = 1./(1. + dm * RiB)**2
        Fh = 1./(1. + dh * RiB)**2


    ustar = np.sqrt( (Am**2)*(U**2)*Fm)
    tstar = 1.0/R * (Ah**2) * U * ( Ta - Ts ) * Fh / ustar
    qstar = 1.0/R * (Ah**2) * Fh * U *(Qa-Qs) / ustar

    Hcan = -1.0 * Constants.Cpd * rho_air * ustar * tstar
    LEC  = -1.0 *  Constants.Lv * rho_air * ustar * qstar

    return LEC, Hcan

def qsat(T,P):
    #------------------------------------------------------------------------
    #  Purpose:
    #     compute saturated specific humidity for given T and P
    #     using Clausius-Clapeyron Equation
    #
    #  Synopsis:
    #     qs=qsat(L,Rv,Rd,T,P)
    #------------------------------------------------------------------------

    #Tref = 298                              # reference temperature @ 25oC
    #eref = 3167                             # reference saturated vapor pressure at Tref
    #es = eref * np.exp( Constants.Lv * ( T - Tref ) / Constants.Rv / T /Tref)
    #rs = (Constants.Rd/Constants.Rv)*es/(P-es)
    #qs = rs/(rs+1)

    def desdT(Tk):
        T  = Tk - 273.15  # convert to Celsius
        es = 611 * np.exp( 17.27 * ( T/(237.3+T) ) )
        dedT = (Constants.Lv * es) / (Constants.Rv * Tk ** 2.0)
        #qs = 0.622 * es / Pair
        return dedT, es
    
    delta, es = desdT(T)
    rs        = (Constants.Rd/Constants.Rv)*es/(P-es)
    qs        = rs/(rs+1)

    return qs

def NetLongwaveRadiation( Ld, TR, TG, TW, TT, TGe, TWe):

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # original name: longwave_2trees_circle
    # % longwave_tree.m: initial version Mar 2014                       %
    # % last updated: 31 Mar 2014                                       %
    # % Author: Young-Hee Ryu                                           %
    # % Compute longwave radiation including urban tree   %
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    LR_total  = np.zeros(shape=(1,pucmparams.nR))
    LG_total  = np.zeros(shape=(1,pucmparams.nG))
    LW_total  = np.zeros(shape=(1,pucmparams.nW))

    LR_total1  = np.zeros(shape=(1,pucmparams.nR))
    LG_total1  = np.zeros(shape=(1,pucmparams.nG))
    LW_total1  = np.zeros(shape=(1,pucmparams.nW))

    LR_total2  = np.zeros(shape=(1,pucmparams.nR))
    LG_total2  = np.zeros(shape=(1,pucmparams.nG))
    LW_total2  = np.zeros(shape=(1,pucmparams.nW))

    for j1 in range(pucmparams.nR):
        LR_total[0,j1]  = pucmparams.eR[j1] * ( Ld - Constants.ss * TR[j1]**4)

    #==========================================================================
    # Sky view factors
    # FGS = 1.0 - pucmparams.FGW - pucmparams.FGT
    # FWS = 1.0 - pucmparams.FWG - pucmparams.FWW - pucmparams.FWT
    # FTS = 1.0 - pucmparams.FTW - pucmparams.FTG - pucmparams.FTT

    # area
    AR = np.array([ pucmparams.W, 2.0 * pucmparams.H, 2.0 * ( 2.0 * np.pi * pucmparams.aleaf ), pucmparams.W ])
    F  = np.array( [  [             0, pucmparams.FGW, pucmparams.FGT ],
                      [ pucmparams.FWG, pucmparams.FWW, pucmparams.FWT ],
                      [ pucmparams.FTG, pucmparams.FTW, pucmparams.FTT ],
                      [ pucmparams.FGS, pucmparams.FWS * AR[1]/AR[3], 0]  ] )
    F[3,2] = 1 - F[3,0] - F[3,1]

    e     = np.array([ pucmparams.eGe, pucmparams.eWe, pucmparams.eT ])
    Lemit = np.array([ pucmparams.eGe * Constants.ss * TGe**4.0, pucmparams.eWe * Constants.ss * TWe**4.0, pucmparams.eT * Constants.ss * TT**4.0, Ld ])

    #--------------------------------------------------------------------------
    # Ground
    #--------------------------------------------------------------------------
    LG_1st_absorption = 0.0
    LG_2nd_absorption = 0.0
    LG_2nd            = np.zeros(4)

    for j in range(4):
        LG_1st_absorption = LG_1st_absorption + Lemit[j] * AR[j] / AR[0] * F[j,0]

    for j in range(4):
        for k in range(3):
            LG_2nd[j] = LG_2nd[j] + AR[j]/AR[k] * F[j,k] * (1.0 - e[k] ) * AR[k] / AR[0] * F[k,0]
        LG_2nd_absorption = LG_2nd_absorption + Lemit[j] * LG_2nd[j]

    # multiple subfacets
    for j3 in range(pucmparams.nG):
        LG_total1[0,j3] = pucmparams.eG[j3] * LG_1st_absorption + LG_2nd_absorption - pucmparams.eG[j3] * Constants.ss * TG[j3]**4

    #--------------------------------------------------------------------------
    # Wall
    #--------------------------------------------------------------------------
    LW_1st_absorption = 0.
    LW_2nd_absorption = 0.
    LW_2nd            = np.zeros(4)

    for j in range(4):
        LW_1st_absorption = LW_1st_absorption + Lemit[j] * AR[j] / AR[1] * F[j,1]

    for j in range(4):
        for k in range(3):
            LW_2nd[j] = LW_2nd[j] + AR[j] / AR[k] * F[j,k] * (1.0 - e[k] ) * AR[k]/AR[1] * F[k,1]
        LW_2nd_absorption = LW_2nd_absorption + Lemit[j] * LW_2nd[j]

    # multiple subfacets
    for j2 in range(pucmparams.nW):
        LW_total1[0,j2] = pucmparams.eW[j2] * LW_1st_absorption + LW_2nd_absorption - pucmparams.eW[j2] * Constants.ss * TW[j2]**4.0

    #--------------------------------------------------------------------------
    # Tree
    #--------------------------------------------------------------------------
    LT_1st_absorption = 0.
    LT_2nd_absorption = 0.
    LT_2nd            = np.zeros(4)

    for j in range(4):
        LT_1st_absorption = LT_1st_absorption + Lemit[j] * AR[j] / AR[2] * F[j,2]

    for j in range(4):
        for k in range(3):
            LT_2nd[j] = LT_2nd[j] + AR[j] / AR[k] * F[j,k] * (1.0 - e[k] ) * AR[k] / AR[2] * F[k,2]
        LT_2nd_absorption = LT_2nd_absorption + Lemit[j] * LT_2nd[j]

    LT_total1 = pucmparams.eT * LT_1st_absorption + LT_2nd_absorption - pucmparams.eT * Constants.ss * TT**4.0

    #==========================================================================
    # without trees
    #==========================================================================
    FGS0 = ( (pucmparams.H/pucmparams.W)**2 + 1.0 )**0.5 - pucmparams.H/pucmparams.W
    FWS0 = 0.5*( pucmparams.H/pucmparams.W + 1.0 - ( (pucmparams.H/pucmparams.W)**2 + 1.0 )**0.5 )/(pucmparams.H/pucmparams.W)
    FWG0 = FWS0
    FGW0 = 1.0 - FGS0
    FWW0 = 1.0 - FWS0 - FWG0

    #%==========================================================================
    # area
    AR = np.array( [ pucmparams.W, 2.0*pucmparams.H, pucmparams.W] )
    F0 = np.array( [ [  0, FGW0             ], 
                    [FWG0, FWW0             ], 
                    [FGS0, FWS0*AR[1]/AR[2] ] ] )

    e     = np.array([ pucmparams.eGe, pucmparams.eWe ])
    Lemit = np.array([ pucmparams.eGe * Constants.ss * TGe**4.0, pucmparams.eWe * Constants.ss * TWe**4.0, Ld])

    #--------------------------------------------------------------------------
    # Ground
    #--------------------------------------------------------------------------
    LG_1st_absorption = 0.
    LG_2nd_absorption = 0.
    LG_2nd            = np.zeros(3)

    for j in range(3):
        LG_1st_absorption = LG_1st_absorption + Lemit[j] * AR[j] / AR[0] * F0[j,0]

    for j in range(3):
        for k in range(2):
            LG_2nd[j] = LG_2nd[j] + AR[j] / AR[k] * F0[j,k] * (1.0 - e[k] ) * AR[k] / AR[0] * F0[k,0]
        LG_2nd_absorption = LG_2nd_absorption + Lemit[j] * LG_2nd[j]

    # multiple subfacets
    for j3 in range(pucmparams.nG):
        LG_total2[0,j3] = pucmparams.eG[j3] * LG_1st_absorption + LG_2nd_absorption - pucmparams.eG[j3] * Constants.ss * TG[j3]**4.0

    #--------------------------------------------------------------------------
    # Wall
    #--------------------------------------------------------------------------
    LW_1st_absorption = 0.
    LW_2nd_absorption = 0.
    LW_2nd            = np.zeros(3)

    for j in range(3):
        LW_1st_absorption = LW_1st_absorption + Lemit[j] * AR[j] / AR[1] * F0[j,1]

    for j in range(3):
        for k in range(2):
            LW_2nd[j] = LW_2nd[j] + AR[j]/AR[k] * F0[j,k] * (1.0 - e[k] ) * AR[k] / AR[1] * F0[k,1]
        LW_2nd_absorption = LW_2nd_absorption + Lemit[j] * LW_2nd[j]

    # multiple subfacets
    for j2 in range(pucmparams.nW):
        LW_total2[0,j2] = pucmparams.eW[j2] * LW_1st_absorption + LW_2nd_absorption - pucmparams.eW[j2] * Constants.ss * TW[j2]**4.0

    Lleaf    = LT_total1 * np.pi / pucmparams.LAItree                             # longwave radiation per unit of leaf plan area
    LT_total = LT_total1 * pucmparams.ftree                                       # total tree longwave radiation
    LG_total = LG_total1 * pucmparams.ftree + LG_total2 * (1.0 - pucmparams.ftree) # total longwave radiation on ground
    LW_total = LW_total1 * pucmparams.ftree + LW_total2 * (1.0 - pucmparams.ftree) # total longwave radiation on the wall

    return LR_total, LG_total, LW_total, LT_total, Lleaf

def SoilInfiltrationRootUptake( inflow, outflow, WC0, D, K, dt, Sroot):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Original name WCdiff_root_uptake_yh
    #% WCdiff.m: initial version Aug 2012                                
    #% Author: Zhihua Wang, Ting Sun                                    
    #% 
    #% Purpose:
    #% Resolve 1D vertical water content diffusion process based on
    #% Richards' equation
    #% 
    #% Syntax:
    #% WCt=WCdiff(inflow,outflow,WC0,b,Hs,nL,dt)
    #% 
    #% Variables:
    #% WCt     - final profile of water content
    #% inflow  - as it is
    #% outflow - as it is 
    #% WC0     - initial profile of water content
    #% b,Hs    - parameters in bc model
    #% nL      - number of discretized layers
    #% dt      - time step used to resolve transport 
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # initialize diffusion water content within layers
    DWG = np.zeros(pucmparams.nL)
    WCt = np.zeros(pucmparams.nL)
    
    # water infiltration through layers
    DWG[0]  = 2.0 * D[0] * ( WC0[0] - WC0[1] )/ ( pucmparams.dgG[0] + pucmparams.dgG[1] ) + K[0]
    DWG[-1] = outflow
    for j in range(1, pucmparams.nL-1):
        DWG[j] = 2.0 * D[j] * ( WC0[j] - WC0[j+1] ) / ( pucmparams.dgG[j] + pucmparams.dgG[j+1] ) + K[j]

    # Young-Hee Ryu
    WCt[0] = WC0[0] + dt * ( inflow - DWG[0] - Sroot[0] ) / pucmparams.dgG[0] 

    for j2 in range(1, pucmparams.nL):
        WCt[j2] = WC0[j2] + dt * (DWG[j2-1] - DWG[j2] - Sroot[j2] ) / pucmparams.dgG[j2]
    
    # WCt: soil moisture/water content
    # DWG: water infiltration through the interface

    return WCt, DWG

def DKeff( W):
    #------------------------------------------------------------------------
    #  Purpose:
    #     compute effective hydraulic diffusivity and conductivity
    #
    #  Synopsis:
    #     [De,Ke]=DKeff(d,W,Ws,Wr,Ks,nL,n,a)
    #
    #  Variable Description:
    #     n,a    - fitting parameters for unsaturated soil
    #     Ws,Wr  - saturated/residual soil water content
    #     Ks     - saturated soil conductivity
    #------------------------------------------------------------------------
    
    D  = np.zeros(shape=(pucmparams.nL)) 
    K  = np.zeros(shape=(pucmparams.nL))
    De = np.zeros(shape=(pucmparams.nL)) 
    Ke = np.zeros(shape=(pucmparams.nL))
        
    # 2. Cosby et al-Chen model (1984)
    b = 5.33 #8.72
    Hs = 0.355  # Saturation soil suction
    for j in range(pucmparams.nL):
        K[j] = pucmparams.Ks * ( W[j] / pucmparams.Ws )**(2*b+3)
        D[j] = b  * pucmparams.Ks * Hs * ( W[j]/pucmparams.Ws )**(b+2)/pucmparams.Ws
    
    for j in range(pucmparams.nL-1):
        Ke[j] = ( pucmparams.dG[j] + pucmparams.dG[j+1] ) / ( ( pucmparams.dG[j] / K[j] ) + ( pucmparams.dG[j+1] / K[j+1] ))    
        De[j] = ( pucmparams.dG[j] + pucmparams.dG[j+1] ) / ( ( pucmparams.dG[j] / D[j] ) + ( pucmparams.dG[j+1] / D[j+1] ))    

    Ke[-1] = K[-1]
    De[-1] = D[-1]

    return De, Ke

def TGF( g, Q, q, i):
    #------------------------------------------------------------------------
    #  Purpose:
    #     compute solid temperatures for walls, roads and roofs
    #  
    #  Synopsis:
    #     [TW,TG,TR]=TGF(gW,gG,gR,QW,QG,QR,qW,qR,T1,T2,T3,TB,i)
    #
    #  Variable Description:
    #     g   - Green's function
    #     Q   - net input heat flux at the expoure surface
    #     q   - heat flux at the inner (building) surface
    #------------------------------------------------------------------------

    S1 = np.trapz( x=g[:(i+1),0], y=np.concatenate(( [0], q[:i][::-1] )) )
    S2 = np.trapz( x=g[:(i+1),1], y=np.concatenate(( [0], Q[:i][::-1] )) )
    q1 = (2.0 * ( S2 - S1 ) + g[1,1] * Q[i] )/ g[1,0]
    S3 = np.trapz( x=g[:(i+1),1], y=np.concatenate(( [0], q[:i][::-1] )) )
    S4 = np.trapz( x=g[:(i+1),0], y=np.concatenate(( [0], Q[:i][::-1] )) )
    T  = -0.5 * q1 * g[1,1] + 0.5 * Q[i] * g[1,0] + (S4 - S3)

    return T, q1

def Green( Fo, dR, kR, alR, ts, n):
    #------------------------------------------------------------------------
    #  Purpose:
    #     compute Green's fucntion for solid layers 
    #  Synopsis:
    #     [g] = Green(Fo,d,k,a,t,n)
    #  Variable Description:
    #     Fo    - Fourier number
    #     d     - thickness
    #     k,a   - thermal conductivity and diffusivity
    #     t     - time
    #------------------------------------------------------------------------
    Fo_cr = 1.0/np.pi/np.sqrt(2)   #characteristic Fourier number
    x     = np.array([ 0, dR])
    nt    = len(ts)
    g     = np.zeros(shape=(nt,2))
    I1 = np.where((Fo<=Fo_cr)&(Fo!=0))
    I2 = np.where( (Fo>Fo_cr)&(Fo!=0))

    if len(I1) > 0:
        # compute small time solution
        R          = np.arange( -math.floor((n-1)/2), math.ceil((n-1)/2) + 1, 1)
        xx, tt, nn = np.meshgrid(x, ts[I1], R)
        #xx = xx.T; tt = tt.T; nn = nn.T;
        K          = np.sqrt( alR * tt / np.pi) * np. exp(-((xx-2 * nn * dR)**2) / 4.0 /alR / tt ) - ( abs(xx-2 * nn * dR)/2.0 ) * special.erfc(abs(xx-2*nn*dR)/2.0/np.sqrt(alR * tt))
        g[I1,:]    = 2.0/kR * K.sum(axis=2)

    if len(I2) > 0:
        # solution based on eigenfunction
        R          =  np.arange( 1, n+1, 1)
        xx, tt, nn = np.meshgrid( x, ts[I2], R)
        K          = np.exp( -alR * ( nn * np.pi / dR )**2.0 * tt )/nn**2.0*np.cos(nn*np.pi*xx/dR)
        xx,tt      = np.meshgrid( x, ts[I2] )
        g[I2,:]    = alR * tt/kR/dR + dR/6./kR*(3*(1-xx/dR)**2-1) - 2*dR/np.pi**2/kR*K.sum(axis=2)

    return g

def NetShortwaveRadiation( Sd, qz, qa, ii):
    """
    original function called shortwave_2trees_circle()
    Sd: downwelling shortwave W/m2
    qz: zenith angle
    qa: 
    ii: index of arrays
    """
    aleaf = pucmparams.aleaf
    htree = pucmparams.htree
    dtree = pucmparams.dtree
    Aleaf = 2 * np.pi * aleaf
    # Transmitance of trees
    tau   = np.exp(-0.61 * pucmparams.LAItree)  # based on Maass et al. (1995), Forest Ecology and Management
    H     = pucmparams.H     # Building height [m]
    W     = pucmparams.W     # Canyon width    [m]
    
    if ( htree + aleaf >= pucmparams.H ):
        htree = htree - 0.000001

    # Solar radiation at each subfacet of the roof
    SR_total = np.zeros(pucmparams.nR)
    for j in range(pucmparams.nR):
        SR_total[j] = (1.0 - pucmparams.aR[j]) * Sd

    # Solar radiation components in the presence of trees
    qs    = qa - pucmparams.qc
    XI    = np.tan(qz) * abs(np.sin(qs))
    secXI = np.sqrt( 1 + XI ** 2 )

    if ( Sd < 0.001 ):
        ST = 0.           # incident solar radiation on tree
        SG = 0.           # incident solar radiation ground (when trees shade)
        SW = 0.           # incident solar radiation wall (when trees shade)
        SW_ftree0 = 0.    # incident solar radiation on wall (no trees)
        SG_ftree0 = 0.    # incident solar radiation on ground (no trees)
    else:
        # Compute reference angles (two tree model)
        XI1 = ( (W-dtree)*(H-htree) + aleaf * np.sqrt( (W-dtree)**2 + (H-htree)**2 - aleaf**2 ) )/( (H-htree)**2 - aleaf**2 )
        XI2 = ( (W-dtree)*(H-htree) - aleaf * np.sqrt( (W-dtree)**2 + (H-htree)**2 - aleaf**2 ) )/( (H-htree)**2 - aleaf**2 )
        XI3 = ( dtree*(H-htree) + aleaf * np.sqrt( dtree**2 + (H-htree)**2 - aleaf**2 ) )/( (H-htree)**2 - aleaf**2 )
        XI4 = ( dtree*(H-htree) - aleaf * np.sqrt( dtree**2 + (H-htree)**2 - aleaf**2 ) )/( (H-htree)**2 - aleaf**2 )
        if ( XI >= XI1 ): # case 1
            # The two trees are completely shaded.
            ST1 = 0.  # ST1 and ST2 are the direct shortwave on trees 1 and 2
            ST2 = 0.
        elif ( (XI >= XI2) & (XI < XI1) ): # case 2
            ST1 = Sd/Aleaf * ( aleaf * secXI + (W-dtree) - (H-htree) * XI )
            ST2 = 0.

        elif ( XI < XI2 ): # case 3
            # ST1 is completely illuminated.
            ST1 = Sd/Aleaf * ( 2*aleaf*secXI )
        
            if ( XI >= XI3 ):  # case 3-1
                # ST2 is completely shaded.
                ST2 = 0.
            elif ( (XI >= XI4) & (XI < XI3) ): #case 3-2
                # ST2 is partially illuminated.
                ST2 = Sd/Aleaf * ( aleaf * secXI + dtree - (H-htree)*XI )
            elif ( XI < XI4 ):  #% case 3-3
                # ST2 is completely illuminated.
                ST2 = Sd/Aleaf * (  2 * aleaf * secXI )
            else: pass

        # Direct solar radiation trees: average two trees
        ST  = ( ST1 + ST2 )/2*(1-tau)
        chi_shaded, eta_shaded, chi_shaded_by_trees, eta_shaded_by_trees = shortwave_shadows_2tree_circle(XI)
        # Direct solar radiation on ground
        SG = Sd / W * ( W - chi_shaded + chi_shaded_by_trees * tau )
        # Direct solar radiation on the wall
        SW = Sd / (2*H) * ( H - eta_shaded + eta_shaded_by_trees * tau ) * XI

        #----------------------------------------------------------------------
        # note by Young-Hee Ryu %
        # Check whether the shortwave radiation is conserved or not.
        # The excess or deficit of the energy is distributed to the tree or wall to conserve the total energy.
        # This is most likely due to the interference between the two trees, 
        # which is not considered here.
        #----------------------------------------------------------------------
        check_total = SG + 2 * H / W * SW + 2 * Aleaf / W * ST

        if ( abs( check_total - Sd ) > 1.e-3 ):
            delta = check_total - Sd
            if ( delta < 0. ):
                ST = ST - delta*W/(2 * Aleaf)
            else:
                SW = SW - delta*W/(2 * H)
            #print(ii, Sd, 'conservation is not met, delta = ', delta,', ST = ', ST,', SW = ',  SW, ii)

        #==========================================================================
        # direct SW without trees
        #==========================================================================
        lsh = H * XI
        if lsh>=W:
            lsh=W
        SW_ftree0 = Sd/(2*H)*lsh
        SG_ftree0 = Sd/W*(W-lsh)
    #==========================================================================
    # Multiple reflection
    # redefine view factors, albedo, direct shortwave radiation using index
    # F(j,i) = Fj->i: from j surface to i surface (absorbed by i surface)
    # % index
    # 0 = ground
    # 1 = wall
    # 2 = tree
    # e.g., F(1,2) = FGW, F(2,3) = FWT
    #==========================================================================

    # Matrix of sky view factors
    F = np.array(  [ [ 0,             pucmparams.FGW, pucmparams.FGT ],
                     [ pucmparams.FWG, pucmparams.FWW, pucmparams.FWT ],
                     [ pucmparams.FTG, pucmparams.FTW, pucmparams.FTT ] ] )
    # Area of surface relative to canyon width (ground, two walls, two trees, sky)
    AR = np.array( [ pucmparams.W, 2 * pucmparams.H, 2 * Aleaf, pucmparams.W])  
    # average albedos (ground, wall, tree)
    a  = np.array([ pucmparams.aGe, pucmparams.aWe, pucmparams.aT])
    # Direct incident solar radiation computed previously
    Sdirect = np.array([ SG, SW, ST])

    #--------------------------------------------------------------------------
    # Ground (solar radiation averaged above all surfaces)
    #--------------------------------------------------------------------------
    SG_1st_reflection = 0.
    SG_2nd_reflection = 0.
    SG_2nd            = np.zeros(3)

    for j in range(3):
        SG_1st_reflection = SG_1st_reflection + a[j] * Sdirect[j] * AR[j] / AR[0] * F[j,0]

    for j in range(3):
        for k in range(3):
            SG_2nd[j] = SG_2nd[j] + AR[j] / AR[k] * F[j,k] * a[k] * AR[k]/AR[0] * F[k,0]
        SG_2nd_reflection = SG_2nd_reflection + a[j] * Sdirect[j] * SG_2nd[j]

    # Total incident solar radiation on the ground: direct + diffuse
    SG_total1 = np.zeros(pucmparams.nG)
    for j3 in range(pucmparams.nG):
        SG_total1[j3] = (1.0 - pucmparams.aG[j3]) * SG + (1.0 - pucmparams.aG[j3] ) * SG_1st_reflection + SG_2nd_reflection

    #--------------------------------------------------------------------------
    # Wall
    #--------------------------------------------------------------------------
    SW_1st_reflection = 0.
    SW_2nd_reflection = 0.
    SW_2nd = np.zeros(3)

    for j in range(3):
        SW_1st_reflection = SW_1st_reflection + a[j] * Sdirect[j] * AR[j] / AR[1] * F[j,1]

    for j in range(3):
        for k in range(3):
            SW_2nd[j] = SW_2nd[j] + AR[j]/AR[k] * F[j,k] * a[k] * AR[k]/AR[1] * F[k,1]
        SW_2nd_reflection = SW_2nd_reflection + a[j] * Sdirect[j] * SW_2nd[j]

    # Total incident solar radiation on the wall: direct + diffuse
    SW_total1 = np.zeros(pucmparams.nW)
    for j2 in range(pucmparams.nW):
        SW_total1[j2] = (1.0 - pucmparams.aW[j2]) * SW + (1.0 - pucmparams.aW[j2] ) * SW_1st_reflection + SW_2nd_reflection

    #--------------------------------------------------------------------------
    # Tree
    #--------------------------------------------------------------------------
    ST_1st_reflection = 0.
    ST_2nd_reflection = 0.
    ST_2nd = np.zeros(3)

    for j in range(3):
        ST_1st_reflection = ST_1st_reflection + a[j] * Sdirect[j] * AR[j] / AR[2] * F[j,2]

    for j in range(3):
        for k in range(3):
            ST_2nd[j] = ST_2nd[j] + AR[j]/AR[k] * F[j,k] * a[k] * AR[k] / AR[2] * F[k,2]
        ST_2nd_reflection = ST_2nd_reflection + a[j] * Sdirect[j] * ST_2nd[j]

    # Total incident solar radiation on trees: direct + diffuse
    ST_total1 = (1.0 - pucmparams.aT )*ST + (1.0-pucmparams.aT) * ST_1st_reflection + ST_2nd_reflection

    #--------------------------------------------------------------------------
    # Reflected radiation into the atmosphere
    # This is not used directly, but in case can be computed to check if the
    # energy is conserved or not.
    #--------------------------------------------------------------------------
    sum_F = F.sum(axis=1)
    # To satisfy the unity rule
    FS = 1 - sum_F
    # %--------------------------------------------------------------------------
    SA_1st_reflection = 0.
    SA_2nd_reflection = 0.
    SA_2nd            = np.zeros(3)

    for j in range(3):
        SA_1st_reflection = SA_1st_reflection + a[j] * Sdirect[j] * AR[j] / AR[3] * FS[j]

    for j in range(3):
        for k in range(3):
            SA_2nd[j] = SA_2nd[j] + a[k] * Sdirect[k] * AR[k] / AR[j] * F[k,j]
        SA_2nd_reflection = SA_2nd_reflection + a[j] * AR[j] / AR[3] * FS[j] * SA_2nd[j]

    SA1 = SA_1st_reflection + SA_2nd_reflection

    #==========================================================================
    # without trees
    #%==========================================================================
    FGS0 = ( (pucmparams.H/pucmparams.W)**2 + 1 )**0.5 - pucmparams.H/pucmparams.W
    FWS0 = 0.5*( pucmparams.H/pucmparams.W + 1. - ( (pucmparams.H/pucmparams.W)**2 + 1 )**0.5 )/(pucmparams.H/pucmparams.W)
    FWG0 = FWS0
    FGW0 = 1.0 - FGS0
    FWW0 = 1.0 - FWS0 - FWG0

    #==========================================================================
    # Multiple reflection
    # redefine view factors, albedo, direct shortwave radiation using index
    # F(j,i) = Fj->i: from j surface to i surface, absorbed by i surface
    # % index
    # 1 = ground
    # 2 = wall
    # e.g., F(1,2) = FGW
    #==========================================================================
    F0  = np.array([ [ 0.0, FGW0 ],  
                     [ FWG0, FWW0]  ])

    # area
    AR = np.array( [ pucmparams.W, 2 * pucmparams.H, pucmparams.W])
    a  = np.array( [ pucmparams.aGe, pucmparams.aWe ])
    Sdirect = np.array( [ SG_ftree0, SW_ftree0 ] )

    #--------------------------------------------------------------------------
    # Ground
    #--------------------------------------------------------------------------
    SG_1st_reflection = 0.
    SG_2nd_reflection = 0.
    SG_2nd = np.zeros(2)

    for j in range(2):
        SG_1st_reflection = SG_1st_reflection + a[j] * Sdirect[j] * AR[j] / AR[0] * F0[j,0]

    for j in range(2):
        for k in range(2):
            SG_2nd[j] = SG_2nd[j] + AR[j]/AR[k]*F0[j,k] * a[k] * AR[k] / AR[0] * F0[k,0]
        SG_2nd_reflection = SG_2nd_reflection + a[j] * Sdirect[j] * SG_2nd[j]

    # multiple subfacets
    SG_total2 = np.zeros(pucmparams.nG)
    for j3 in range(pucmparams.nG):
        SG_total2[j3] = (1.0 - pucmparams.aG[j3] ) * SG_ftree0 + (1.0 - pucmparams.aG[j3] ) * SG_1st_reflection + SG_2nd_reflection
 
    #--------------------------------------------------------------------------
    # Wall
    #--------------------------------------------------------------------------
    SW_1st_reflection = 0.
    SW_2nd_reflection = 0.
    SW_2nd = np.zeros(2)

    for j in range(2):
        SW_1st_reflection = SW_1st_reflection + a[j]*Sdirect[j]*AR[j]/AR[1] * F0[j,1]

    for j in range(2):
        for k in range(2):
            SW_2nd[j] = SW_2nd[j] + AR[j]/AR[k] * F0[j,k] * a[k] * AR[k]/AR[1] * F0[k,1]
        SW_2nd_reflection = SW_2nd_reflection + a[j] * Sdirect[j] * SW_2nd[j]

    # multiple subfacets
    SW_total2 = np.zeros(pucmparams.nW)
    for j2 in range(pucmparams.nW):
        SW_total2[j2] = (1.0 - pucmparams.aW[j2] ) * SW_ftree0 + (1.0 - pucmparams.aW[j2] ) * SW_1st_reflection + SW_2nd_reflection
 
    #--------------------------------------------------------------------------
    # Reflected radiation into the atmosphere
    # This is not used directly, but in case can be computed to check if the
    # energy is conserved or not.
    #--------------------------------------------------------------------------
    sum_F0 = F0.sum(axis=1)
    # To satisfy the unity rule
    FS0 = 1.0 - sum_F0
    #--------------------------------------------------------------------------
    SA_1st_reflection = 0.
    SA_2nd_reflection = 0.
    SA_2nd = np.zeros(2)

    for j in range(2):
        SA_1st_reflection = SA_1st_reflection + a[j] * Sdirect[j] * AR[j] / AR[2] * FS0[j]

    for j in range(2):
        for k in range(2):
            SA_2nd[j] = SA_2nd[j] + a[k] * Sdirect[k] * AR[k]/AR[j]*F0[k,j]
        SA_2nd_reflection = SA_2nd_reflection + a[j]*AR[j]/AR[2] * FS0[j] * SA_2nd[j]

    # Compute net shortwave radiation above each subfacet (average with/out tree areas)
    #SA2      = SA_1st_reflection + SA_2nd_reflection
    Sleaf    = ST_total1 * Aleaf/(pucmparams.LAItree * 2 * aleaf)                  # 
    ST_total = ST_total1 * pucmparams.ftree                                        # solar radiation on trees
    SG_total = SG_total1 * pucmparams.ftree + SG_total2 * (1.0 - pucmparams.ftree)  # solar radiation on ground
    SW_total = SW_total1 * pucmparams.ftree + SW_total2 * (1.0 - pucmparams.ftree)  # solar radiation on wall
    #SA       = SA1*pucmparams.ftree + SA2 * (1.0 - pucmparams.ftree)                # 
    
    return SR_total, SG_total, SW_total, ST_total, Sleaf

def shortwave_shadows_2tree_circle(XI):
    H = pucmparams.H
    W = pucmparams.W
    
    chi_wall = H * XI
    eta_wall = H - W/XI
    if ( chi_wall < W ):  # XI < W/H
        eta_wall = 0
    else:                 # XI >= W/H
        chi_wall = W
    ## the origin (0,0) is the lower left corner of the canyon
    x0 = max(0., W - H * XI)
    y0 = max(0., H - W/XI)
    secXI   = np.sqrt( 1. + XI**2 )
    cosecXI = np.sqrt ( 1. + 1./(XI**2) )
    #---------------------------
    # Shadow by the Tree 1
    #---------------------------
    x1 = max( 0, pucmparams.dtree - pucmparams.htree * XI - pucmparams.aleaf * secXI)
    y1 = max( 0, pucmparams.htree - (W-pucmparams.dtree)/XI - pucmparams.aleaf * cosecXI)
    x2 = max( 0, pucmparams.dtree - pucmparams.htree * XI + pucmparams.aleaf * secXI)
    y2 = max( 0, pucmparams.htree - (W-pucmparams.dtree)/XI + pucmparams.aleaf * cosecXI)
    chi_tree1 = x2 - x1
    #---------------------------
    # Shadow by the Tree 2
    #---------------------------
    x3 = max( 0, W - pucmparams.dtree - pucmparams.htree * XI - pucmparams.aleaf * secXI)
    y3 = max( 0, pucmparams.htree - pucmparams.dtree/XI - pucmparams.aleaf * cosecXI)
    x4 = max( 0, W - pucmparams.dtree - pucmparams.htree * XI + pucmparams.aleaf * secXI)
    y4 = max( 0, pucmparams.htree - pucmparams.dtree/XI + pucmparams.aleaf * cosecXI)
    chi_tree2 = x4 - x3
    eta_tree1 = y4 - y3
    eta_tree2 = y2 - y1
    #-----------------------------------------------------------------
    # Total shadow length on the ground by Wall, Tree 1, and Tree 2
    #-----------------------------------------------------------------
    delta = max([0, x2-x0])

    if ( x0 < x4 ):
        chi_shaded = W - min([x0, x3]) + chi_tree1 - delta
        if ( x0 < x3 ):
            chi_shaded_by_trees = chi_tree1 - delta
        else:
            chi_shaded_by_trees = chi_tree1 + x0 - x3
    elif ( x0 >= x4 ):
        chi_shaded          = chi_wall + chi_tree1 + chi_tree2
        chi_shaded_by_trees =  chi_tree1 + chi_tree2
    else:
        pass

    #-----------------------------------------------------------------
    # Total shadow length on the wall by Wall, Tree 1, and Tree 2
    #-----------------------------------------------------------------
    lowest_shaded = max([y0, y2])

    if ( y3 > lowest_shaded ):
        eta_shaded = eta_tree1 + lowest_shaded
        if ( y2 > eta_wall ):
            eta_shaded_by_trees = eta_tree1 + y2 - eta_wall
        elif ( y2 <= eta_wall ):
            eta_shaded_by_trees = eta_tree1
        elif ( y1 > eta_wall ):
            eta_shaded_by_trees = eta_tree1 + eta_tree2
    else:
        eta_shaded = max([y0, y1, y2, y3, y4])
        if ( y4 > eta_wall ):
            eta_shaded_by_trees = y4 - eta_wall
        else:
            eta_shaded_by_trees = 0.
    
    return chi_shaded, eta_shaded, chi_shaded_by_trees, eta_shaded_by_trees

def Zenith(df, offset, Lat, Lon):
    # Prathap_Zenith_Azimuth
    # This program calculates the solar zenith and azimuth angle for any day
    # The calculations are based on NOAA
    # Refer http://www.jgiesen.de/astro/suncalc/calculations.htm
    # the inputs for this program are
    # Lat - Latitude in degrees
    # Lon - Longitude in degress
    # jday - Julian day 
    # nofdays - number of days
    # offset_time in hours= +/- offset time (eg -4 for EDT and +5.5 for IST) 
    #Lat  = pucmparams.Lat
    #Lon  = pucmparams.Lon
    doy  = np.array(df.index.day_of_year)
    hour = np.array(df.index.hour)
    min  = np.array(df.index.minute)
    sec  = np.array(df.index.second)
    hour = hour + min/60 + sec/3600 
    y    = (2.0 * np.pi/365)*( doy - 1 + (hour-12)/24)
    eqtime = 229.18 * ( 0.000075 + 0.001868 * np.cos(y) - 0.032077 * np.sin(y) - 0.014615 * np.cos(2*y) - 0.040849 * np.sin( 2 * y ))  # in minutes
    declin = 0.006918-0.399912 * np.cos(y) + 0.070257 * np.sin(y) - 0.006758 * np.cos(2*y) + 0.000907 * np.sin(2*y) - 0.002697 * np.cos(3*y) + 0.00148 * np.sin(3*y)
    # offset is the difference between local and utc, so EDT is +4
    time_offset = eqtime - 4 * (Lon) + 60 * offset
    tst = hour * 60  + time_offset #+ min
    ha  = tst/4 - 180
    zenithnew = np.arccos(np.sin(Lat * np.pi/180) * np.sin(declin) + np.cos(Lat * np.pi/180) * np.cos(declin)*np.cos(ha * np.pi/180))
    azimuthnew= np.arccos(-(np.sin(Lat * np.pi/180) * np.cos(zenithnew) - np.sin(declin))/(np.cos(Lat * np.pi/180)*np.sin(zenithnew)))
    zenithnew[ zenithnew >= np.pi/2. ] = np.pi/2.
    for i in range(len(ha)):
        if ha[i] >= 0.:
            azimuthnew[i] = 2*np.pi - azimuthnew[i]
    
    return zenithnew, azimuthnew

def updateRunoffTemp( Trunoffold, h_new, h_old, Tground, Tsurf, meanU, rain_rate, statLayer, SurfaceLenght, dt):
    # Implicit method
    dz    = h_new/2   
    kw    = Constants.rW * Constants.ch2o * pucmparams.DWater_Molecular
    alpha = kw/(Constants.rW * Constants.ch2o * dz)
    
    # heat input from runoff
    qin  = Constants.rW * rain_rate
    qout = Constants.rW * meanU * ((h_new-statLayer)/SurfaceLenght)
    
    bottom    = 1 + 2*alpha*dt/h_new + dt * qout/(h_new * Constants.rW)
    Trunnew_0 = Trunoffold * h_old / h_new + (alpha * dt / h_new) * (Tground + Tsurf) + ((dt * qin)/(h_new * Constants.rW)) *  Tsurf
    Trunnew   = Trunnew_0/bottom
    return Trunnew

def updateSurfaceWaterTemperature(T_air, rho_air, Tcan, h_old, Trunoff, prec_rate, LE, Ce, Rlw, meanU, Train):
    # Energy budget at the top of runoff -------
    # Rlw + LE + Qr + H + Qwt = 0
    
    # dz: this is half the runoff depth, [m]
    dz  = h_old/2   
    kw  = Constants.rW * Constants.ch2o * pucmparams.DWater_Molecular 
    
    # Rain temperature
    #Train = T_air
    
    # Solve for temperature
    A = prec_rate * Constants.rW * Constants.ch2o
    
    if Ce <= 0:
        Ce = 0.1
    if Tcan <= 0:
        Tcan = T_air
        
    B   = Ce * meanU * rho_air * Constants.Cpd
    C   = kw / dz
    Tws = (Rlw - LE + A * Train + B * Tcan + C * Trunoff ) / (A + B + C)
    
    return Tws

def SEBHeatEquation(Tg, Rsw, dt, Trunoff, hmean, Minv, K_cond, dz_ground, K_diff):
    # It does not take evaporation into account
    
    # Solve energy budget -------------------------------------------------
    dz         = hmean/2
    kw         = Constants.rW * Constants.ch2o * pucmparams.DWater_Molecular 
    Qwb        = kw * Trunoff / dz
    G          =  K_cond * Tg[1] / dz_ground
    Tgsurf_new = ( Rsw + Qwb + G )/( kw/dz + K_cond/dz_ground )
    
    # Solve heat equation -------------------------------------------------
    #Tg[0]    = Tgsurf_new
    #Tg[1:-2] = Tg[1:-2] + dt * pucmparams.alg[0] * ( Tg[2:-1] - 2 * Tg[1:-2] + Tg[0:-3] )/ ( pucmparams.dz_ground * pucmparams.dz_ground )
    #Tg[-1]   = Tg[-2]

    # Backward Euler
    Fo          = K_diff * dt / ( dz_ground * dz_ground )
    Tg[0]       = Tgsurf_new
    T_new       = np.copy(Tg)
    T_new[1]    = T_new[1] + Tgsurf_new * Fo
    T_new[1:-1] = np.dot( Minv, T_new[1:-1])
    Tg          = np.copy(T_new)
    
    return Tg

def SEBHeatEquationNoRain( Tg, dt, Minv, qg, dz_ground, K_cond, K_diff):

    # Solve energy budget at the surface
    Tgsurf_new  = Tg[1] + ( qg * dz_ground / K_cond )

    Fo          = K_diff * dt / ( dz_ground * dz_ground )
    Tg[0]       = Tgsurf_new
    T_new       = np.copy(Tg)
    T_new[1]    = T_new[1] + Tgsurf_new * Fo
    T_new[1:-1] = np.dot( Minv, T_new[1:-1])
    Tg          = np.copy(T_new)

    return Tg
    
def StabilityEffects( ind, Tair, Tsurf, Uair, hlayer):
    # From https://link.springer.com/article/10.1007/s10546-010-9523-y

    z0  = pucmparams.ZmG[ind]
    z0h = pucmparams.ZhG[ind]

    if (ind <= 1):
        if hlayer > pucmparams.G_StLayer[ind]:
            z0  = 10**-5
            z0h = z0
        elif (hlayer < pucmparams.G_StLayer[ind]) and (hlayer > pucmparams.G_hc0[ind]):
            C      = hlayer/pucmparams.G_StLayer[ind]
            z0_asp = pucmparams.dwG        # roughness of asphalt [m]
            z0_wat = 10**-5               # roughness of water
            z0_e   = C * np.log(z0_wat) + (1-C) * np.log(z0_asp)
            z0     = np.exp(z0_e)
            z0h    = z0
        else:
            z0  = pucmparams.ZmG[ind]
            z0h = pucmparams.ZhG[ind]

    zref = pucmparams.d
    F2   = (1.0 - z0 / zref)**2.0 / (1.0 - z0h/zref)
    RiB  = Constants.g * zref * (Tair - Tsurf) * F2 / ( (Tair + Tsurf) * 0.5 * Uair ** 2)

    alpha = np.log(zref/z0)
    beta  = np.log(z0/z0h)

    if RiB < 0:   # Unstable atmosphere
        a_u11 = 0.0450
        b_u11 = 0.0030
        b_u12 = 0.0059
        b_u21 = -0.0828
        b_u22 = 0.8845
        b_u31 = 0.1739
        b_u32 = -0.9213
        b_u33 = -0.1057
        part1 = ( b_u11 * beta + b_u12 ) * alpha**2 + (b_u21 * beta + b_u22)*alpha + (b_u31 * beta**2 + b_u32 * beta + b_u33)
        zeta  = a_u11 * alpha * RiB ** 2 + part1 * RiB
    elif (0 < RiB <= 0.2): # Weakly stable atmosphere
        a_w11 = 0.5738
        a_w12 = -0.4399
        a_w21 = -4.901
        a_w22 = 52.50
        b_w11 = -0.0539
        b_w12 = 1.540
        b_w21 = -0.6690
        b_w22 = -3.282
        part1 = ( ( a_w11 * beta + a_w12 ) * alpha + ( a_w21 * beta + a_w22 ) ) * RiB ** 2
        part2 = ( ( b_w11 * beta + b_w12 ) * alpha + ( b_w21 * beta + b_w22 ) ) * RiB
        zeta  = part1 + part2
    elif RiB > 0.2:
        a_s11 = 0.7529
        a_s21 = 14.94
        b_s11 = 0.1569
        b_s21 = -0.3091
        b_s22 = -1.303
        zeta  = ( a_s11 * alpha + a_s21) * RiB + b_s11 * alpha + b_s21 * beta + b_s22
    else: pass

    # Compute stability dependence functions
    if zeta < 0:
        if zeta < -30:
            zeta = -30
        xx        = (1 - 16 * zeta) ** (-1/2)
        psi_m     = 2 * np.log( (1+xx)/2 ) + np.log( (1+xx**2)/2) - 2 * np.arctan(xx) + np.pi/2
        psi_s     = 2 * np.log( (1 + xx**2) / 2)

    else:
        if zeta > 1:
            zeta = 1
        psi_m    = -5 * zeta
        psi_s    = -5 * zeta

    # Assuming a Prandtl number of 1
    # For scalars ------------------------------
    Ch_top    = Constants.kappa ** 2.0  
    Ch_bot    = ( np.log(zref/z0) - psi_m ) * ( np.log(zref/z0h) - psi_s )
    Ch        = Ch_top/Ch_bot
    # For momentum -----------------------------
    Cm        = Constants.kappa ** 2.0 / ( np.log(zref/z0) - psi_m ) ** 2.0
    ustar     = np.sqrt( Cm * Uair ** 2.0 )

    return Ch, Cm, ustar, RiB
    
def TurbulentFluxesGroundPM( ind, hlayer, rho_air, meanU, qair, Tsurf, Tair, Pair):

    Ce, Cm, ustar, RiB = StabilityEffects( ind, Tair, Tsurf, meanU, hlayer)
        
    H   = Ce * rho_air * Constants.Cpd * meanU * ( Tsurf - Tair)              # W/m2
    # ------------------------------------------
    def desdT(Tk):
        T    = Tk - 273.15  # convert to Celsius
        es   = 611 * np.exp( 17.27 * ( T/(237.3+T) ) )
        dedT = (Constants.Lv * es) / (Constants.Rv * Tk ** 2.0)
        #qs = 0.622 * es / Pair
        return dedT, es
    
    ea            = (qair * rho_air) * Constants.Rv * Tair
    delta, es_air = desdT(Tair)
    gamma         = (Constants.Cpd * Pair)/(0.622 * Constants.Lv)
    Ea            = ( 1.25 * 10**-8 * meanU ) * (es_air - ea) * Constants.Lv # W/m2
    LE            = (delta/gamma) * H + Ea

    rs = (Constants.Rd/Constants.Rv)*es_air/(Pair-es_air)
    qs = rs/(rs+1)

    # ------------------------------------------
    tstar = ( H / (rho_air * Constants.Cpd ) )/ ustar
    qstar = ( LE / (Constants.Lv * rho_air ) )/ ustar

    return ustar, LE, H, qstar, tstar, RiB, Ce, qs
    
def TurbulentFluxesGround( ind, qsat, hlayer, rho_air, meanU, qair, Tsurf, Tair ):

    Ce, Cm, ustar = StabilityEffects( ind, Tair, Tsurf, meanU, hlayer)
    
    LE  = Ce * Constants.Lv * rho_air * meanU * ( qsat - qair )  # W/m2 
    H   = Ce * rho_air * Constants.Cpd * meanU * ( Tsurf - Tair) # W/m2
    qstar = ( LE / (Constants.Lv * rho_air ) )/ ustar
    tstar = ( H / (rho_air * Constants.Cpd ) )/ ustar
    return ustar, LE, H, qstar, tstar

def StabilityEffectsRoof( ind, Tair, Tsurf, Uair, hlayer):
    # From https://link.springer.com/article/10.1007/s10546-010-9523-y
    
    if hlayer > pucmparams.R_StLayer[ind]:
        z0  = 10**-5
        z0h = z0
    elif (hlayer < pucmparams.R_StLayer[ind]) and (hlayer > pucmparams.R_hc0[ind]):
        C      = hlayer/pucmparams.R_StLayer[ind]
        z0_asp = pucmparams.dwG        # roughness of asphalt [m]
        z0_wat = 10**-5               # roughness of water
        z0_e   = C * np.log(z0_wat) + (1-C) * np.log(z0_asp)
        z0     = np.exp(z0_e)
        z0h    = z0
    else:
        z0  = pucmparams.ZmR[ind]
        z0h = pucmparams.ZhR[ind]

    zref = pucmparams.Za - pucmparams.Zr
    F2   = (1.0 - z0 / zref)**2.0 / (1.0 - z0h/zref)
    RiB  = Constants.g * zref * (Tair - Tsurf) * F2 / ( (Tair + Tsurf) * 0.5 * Uair ** 2)

    alpha = np.log(zref/z0)
    beta  = np.log(z0/z0h)

    if RiB < 0:   # Unstable atmosphere
        a_u11 = 0.0450
        b_u11 = 0.0030
        b_u12 = 0.0059
        b_u21 = -0.0828
        b_u22 = 0.8845
        b_u31 = 0.1739
        b_u32 = -0.9213
        b_u33 = -0.1057
        part1 = ( b_u11 * beta + b_u12 ) * alpha**2 + (b_u21 * beta + b_u22)*alpha + (b_u31 * beta**2 + b_u32 * beta + b_u33)
        zeta  = a_u11 * alpha * RiB ** 2 + part1 * RiB
    elif (0 < RiB <= 0.2): # Weakly stable atmosphere
        a_w11 = 0.5738
        a_w12 = -0.4399
        a_w21 = -4.901
        a_w22 = 52.50
        b_w11 = -0.0539
        b_w12 = 1.540
        b_w21 = -0.6690
        b_w22 = -3.282
        part1 = ( ( a_w11 * beta + a_w12 ) * alpha + ( a_w21 * beta + a_w22 ) ) * RiB ** 2
        part2 = ( ( b_w11 * beta + b_w12 ) * alpha + ( b_w21 * beta + b_w22 ) ) * RiB
        zeta  = part1 + part2
    elif RiB > 0.2:
        a_s11 = 0.7529
        a_s21 = 14.94
        b_s11 = 0.1569
        b_s21 = -0.3091
        b_s22 = -1.303
        zeta  = ( a_s11 * alpha + a_s21) * RiB + b_s11 * alpha + b_s21 * beta + b_s22
    else: pass

    # Compute stability dependence functions
    if zeta < 0:
        if zeta < -30:
            zeta = -30
        xx        = (1 - 16 * zeta) ** (-1/2)
        psi_m     = 2 * np.log( (1+xx)/2 ) + np.log( (1+xx**2)/2) - 2 * np.arctan(xx) + np.pi/2
        psi_s     = 2 * np.log( (1 + xx**2) / 2)

    else:
        if zeta > 1:
            zeta = 1
        psi_m    = -5 * zeta
        psi_s    = -5 * zeta

    # Assuming a Prandtl number of 1
    # For scalars ------------------------------
    Ch_top    = Constants.kappa ** 2.0  
    Ch_bot    = ( np.log(zref/z0) - psi_m ) * ( np.log(zref/z0h) - psi_s )
    Ch        = Ch_top/Ch_bot
    # For momentum -----------------------------
    Cm        = Constants.kappa ** 2.0 / ( np.log(zref/z0) - psi_m ) ** 2.0
    ustar     = np.sqrt( Cm * Uair ** 2.0 )

    return Ch, Cm, ustar, RiB

def TurbulentFluxesRoofPM( ind, hlayer, rho_air, meanU, qair, Tsurf, Tair, Pair ):

    Ce, Cm, ustar, RiB = StabilityEffectsRoof( ind, Tair, Tsurf, meanU, hlayer)
        
    H   = Ce * rho_air * Constants.Cpd * meanU * ( Tsurf - Tair)              # W/m2

    # ------------------------------------------
    def desdT(Tk):
        T  = Tk - 273.15  # convert to Celsius
        es = 611 * np.exp( 17.27 * ( T/(237.3+T) ) )
        dedT = (Constants.Lv * es) / (Constants.Rv * Tk ** 2.0)
        #qs = 0.622 * es / Pair
        return dedT, es
    
    ea            = (qair * rho_air) * Constants.Rv * Tair
    delta, es_air = desdT(Tair)
    gamma         = (Constants.Cpd * Pair)/(0.622 * Constants.Lv)
    Ea            = ( 1.25 * 10**-8 * meanU ) * (es_air - ea) * Constants.Lv # W/m2
    LE            = (delta/gamma) * H + Ea

    # ------------------------------------------
    tstar = ( H / (rho_air * Constants.Cpd ) )/ ustar
    qstar = ( LE / (Constants.Lv * rho_air ) )/ ustar
    rs    = (Constants.Rd/Constants.Rv)*es_air/(Pair-es_air)
    qs    = rs/(rs+1)

    return ustar, LE, H, qstar, tstar, RiB, Ce, qs

# --------------------------------------------------------
# Parameterizations of turbulent fluxes ------------------

def TurbulentFluxesWallSLUCM(rho_air, CE, U, TW, Tcan):
    H = rho_air * Constants.Cpd * CE * U * ( TW - Tcan )
    return H

def TurbulentFluxesGroundSLUCM(rho_air, CE, U, Tg, Tcan, qcan, qs):
    # Sensible heat flux
    H  = rho_air * Constants.Cpd * CE * U * ( Tg - Tcan )
    # Latent heat flux
    LE    = rho_air * Constants.Lv  * CE * U * ( qs - qcan)
    ustar = 0
    return LE, H, ustar

def TurbulentFluxesCanyonAero( Ta, Qa, rho_air, Ts, Qs, U, zref, z0m, z0h):
    #------------------------------------------------------------------------
    #  Purpose:
    #     compute u*, t*, q* to calculate sensible/latent heat flux
    #     Based on Kot and Song (1998, BLM)
    #
    #  Variable Description:
    #     U   - wind speed
    #     Ta  - potential temperature in K of atmosphere
    #     Ts  - temperature in K at surface
    #     q   - water content of atmosphere/surface
    #     z0m - aerodynamic roughness for momentum
    #     z0h - aerodynamic roughness for heat
    #------------------------------------------------------------------------
    g  = 9.81           # gravity constant
    k  = 0.4            # Von-Karman constants
    bm = 8.0
    bh = 23.0
    R  = 1.0
    F2 = (1.0 - z0m / zref )**2/(1.0 - z0h / zref)
    RiB = g * zref * ( Ta - Ts ) * F2 / ( (Ta + Ts)*0.5*U**2)

    Am  = k/np.log(zref/z0m)
    Ah  = k/np.sqrt(np.log(zref/z0m)*np.log(zref/z0h))

    if ( RiB < 0. ): # unstable condition
        C1 = -0.9848
        C2 = 2.5398 
        C3 = -0.2325
        C4 = 14.1727
        Cmstar = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
    
        if ( (z0m/z0h >= 1.0) & (z0m/z0h <= 100.) ):
            C1 = -1.1790
            C2 = -1.9256
            C3 =  0.1007
            C4 = 16.6796
        elif ( (z0m/z0h > 100.) & (z0m/z0h <= 10000.) ):
            C1 = -1.0487
            C2 = -1.0689
            C3 =  0.0952
            C4 = 11.7828
    
        Chstar = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
    
        cm = Cmstar * Am**2 * bm * np.sqrt(F2) *( (zref/z0m)**(1./3.) - 1.0 )**(3./2.)
        ch = Chstar * Ah**2 * bh * np.sqrt(F2) *( (zref/z0h)**(1./3.) - 1.0 )**(3./2.)
        Fm = 1.0 - (bm*RiB)/(1. + cm * np.sqrt( abs(RiB)) )
        Fh = 1.0 - (bh*RiB)/(1. + ch * np.sqrt( abs(RiB)) )
    
    else:
        if ( (z0m/z0h >= 1.0) & (z0m/z0h <= 100.) ):
            C1 = -0.4738
            C2 = -0.3268
            C3 =  0.0204
            C4 = 10.0715
        elif ( (z0m/z0h > 100.) & (z0m/z0h <= 10000.) ):
            C1 = -0.4613
            C2 = -0.2402
            C3 =  0.0146
            C4 =  8.9172
        else: pass

        dm = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
    
        if ( (z0m/z0h >= 1.0) & (z0m/z0h <= 100.) ):
            C1 =  -0.5128
            C2 =  -0.9448
            C3 =   0.0643
            C4 =  10.8925
        elif ( (z0m/z0h > 100.) & (z0m/z0h <= 10000.) ):
            C1 = -0.3169
            C2 = -0.3803
            C3 =  0.0205
            C4 =  7.5213
        else: pass

        dh = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
    
        Fm = 1./(1. + dm * RiB)**2
        Fh = 1./(1. + dh * RiB)**2


    ustar = np.sqrt( (Am**2)*(U**2)*Fm)
    tstar = 1.0/R * (Ah**2) * U * ( Ta - Ts ) * Fh / ustar
    qstar = 1.0/R * (Ah**2) * Fh * U *(Qa-Qs) / ustar

    Hcan = -1.0 * Constants.Cpd * rho_air * ustar * tstar
    LEC  = -1.0 *  Constants.Lv * rho_air * ustar * qstar

    return LEC, Hcan, ustar

def TurbulentFluxesRoofGroundAero( Ta, Qa, Qs, rho_air, Ts, U, zref, z0m, z0h):

    g  = 9.81           # gravity constant
    k  = 0.4            # Von-Karman constants
    bm = 8.0
    bh = 23.0
    R  = 1.0
    F2 = (1.0 - z0m / zref )**2/(1.0 - z0h / zref)
    RiB = g * zref * ( Ta - Ts ) * F2 / ( (Ta + Ts)*0.5*U**2)

    Am  = k/np.log(zref/z0m)
    Ah  = k/np.sqrt(np.log(zref/z0m)*np.log(zref/z0h))

    if ( RiB < 0. ): # unstable condition
        C1 = -0.9848
        C2 = 2.5398 
        C3 = -0.2325
        C4 = 14.1727
        Cmstar = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
    
        if ( (z0m/z0h >= 1.0) & (z0m/z0h <= 100.) ):
            C1 = -1.1790
            C2 = -1.9256
            C3 =  0.1007
            C4 = 16.6796
        elif ( (z0m/z0h > 100.) & (z0m/z0h <= 10000.) ):
            C1 = -1.0487
            C2 = -1.0689
            C3 =  0.0952
            C4 = 11.7828
    
        Chstar = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
    
        cm = Cmstar * Am**2 * bm * np.sqrt(F2) *( (zref/z0m)**(1./3.) - 1.0 )**(3./2.)
        ch = Chstar * Ah**2 * bh * np.sqrt(F2) *( (zref/z0h)**(1./3.) - 1.0 )**(3./2.)
        Fm = 1.0 - (bm*RiB)/(1. + cm * np.sqrt( abs(RiB)) )
        Fh = 1.0 - (bh*RiB)/(1. + ch * np.sqrt( abs(RiB)) )
    
    else:
        if ( (z0m/z0h >= 1.0) & (z0m/z0h <= 100.) ):
            C1 = -0.4738
            C2 = -0.3268
            C3 =  0.0204
            C4 = 10.0715
        elif ( (z0m/z0h > 100.) & (z0m/z0h <= 10000.) ):
            C1 = -0.4613
            C2 = -0.2402
            C3 =  0.0146
            C4 =  8.9172
        else: 
            pass
        dm = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
    
        if ( (z0m/z0h >= 1.0) & (z0m/z0h <= 100.) ):
            C1 =  -0.5128
            C2 =  -0.9448
            C3 =   0.0643
            C4 =  10.8925
        elif ( (z0m/z0h > 100.) & (z0m/z0h <= 10000.) ):
            C1 = -0.3169
            C2 = -0.3803
            C3 =  0.0205
            C4 =  7.5213
        else: pass

        dh = C1 * np.log(zref/z0m) + C2 * np.log(z0m/z0h) + C3 * np.log(zref/z0m) * np.log(z0m/z0h) + C4
    
        Fm = 1./(1. + dm * RiB)**2
        Fh = 1./(1. + dh * RiB)**2


    ustar = np.sqrt( (Am**2)*(U**2)*Fm)
    tstar = 1.0/R * (Ah**2) * U * ( Ta - Ts ) * Fh / ustar
    qstar = 1.0/R * (Ah**2) * Fh * U *(Qa-Qs) / ustar

    Hrof   = -1.0 * Constants.Cpd * rho_air * ustar * tstar
    LErof  = -1.0 *  Constants.Lv * rho_air * ustar * qstar
    CE     = Hrof / ( Constants.Cpd * rho_air * U * ( Ts - Ta ) )
    return LErof, Hrof, CE, ustar

def MOST( zeta, zeta_scal):
    
    if zeta < 0:
        if zeta < -5:
            zeta      = -5
            zeta_scal = -5
        xx        = (1 - 16 * zeta) ** (1/4)
        psi_m     = 2 * np.log( (1+xx)/2 ) + np.log( (1+xx**2)/2) - 2 * np.arctan(xx) + np.pi/2
        xx        = (1 - 16 * zeta_scal) ** (1/4)
        psi_s     = 2 * np.log( (1 + xx**2) / 2)
    else:
        if 0 <= zeta < 1:
            psi_m    = -5 * zeta
            psi_s    = -5 * zeta_scal
        else:
            zeta = 1
            zeta_scal = 1
            psi_m   = -5 -5 * np.log(zeta)
            psi_s   = -5 -5 * np.log(zeta_scal)
    return psi_m, psi_s

def TurbulentFluxesMOST( z2, z1, z1_h, u2, u1, T2, T1, Q2, Q1, L, rho_air, facet='notwall'):
    
    if facet == 'notwall':
        if abs(L) > 0.05: 
            zeta1_m           = z1 / L
            zeta1_s           = z1_h / L
            zeta2_m           = z2 / L
            zeta2_s           = z2 / L
            psi_m1, psi_s1    = MOST(zeta1_m, zeta1_s)
            psi_m2, psi_s2    = MOST(zeta2_m, zeta2_s)
        else:
            zeta1_m = z1
            zeta2_m = z2
            zeta1_s = z1_h
            zeta2_s = z2
            psi_m2 = psi_m1 = psi_s1 = psi_s2 = 0
            
        ustar = (u2 - u1) * Constants.kappa / (np.log(zeta2_m/zeta1_m) - psi_m2 + psi_m1)
        tstar = (T1 - T2) * Constants.kappa / (np.log(zeta2_s/zeta1_s) - psi_s2 + psi_s1)
        qstar = (Q1 - Q2) * Constants.kappa / (np.log(zeta2_s/zeta1_s) - psi_s2 + psi_s1)
    
        if abs(u2 * ( T1 - T2 )) > 0.0001:
            CE    = tstar * ustar  / ( u2 * ( T1 - T2 ) )
        else:
            CE    = 0.001

        H   = rho_air * Constants.Cpd * tstar * ustar 
        LE  = rho_air * Constants.Lv  * qstar * ustar
        return LE, H, CE, ustar
    
    elif facet == 'wall':
        ustar = (u2 - u1) * Constants.kappa / ( np.log(z2/z1) )
        tstar = (T1 - T2) * Constants.kappa / ( np.log(z2/z1_h) )
        H     = rho_air * Constants.Cpd * tstar * ustar 
        return H

def RunoffRateHERB( vol, rain, dt, qLold, n, Le, L, s0):
    top =  - qLold * dt/2 - (2./5.) * (n/np.sqrt(s0))**(3./5.) * ( L - (3/8.) * Le ) * qLold**(3./5.)
    top = vol + rain * L * dt + top
    bot = (3./5.) * (n/np.sqrt(s0))**(3/5)
    if qLold > 0:
        bot = dt/2 + bot * ( L - (3./8.) * Le ) * qLold**(-2./5.)
    else:
        bot = dt/2 
    qLnew = top/bot
    return qLnew

# PUCM optimization

# Function to compute cross-correlation and find optimal time shift

def find_optimal_time_shift(df, col1, col2, max_shift):
    cross_corr = []
    for shift in range(-max_shift, max_shift + 1):
        shifted_col2 = df[col2].shift(shift)
        correlation = df[col1].corr(shifted_col2)
        cross_corr.append((shift, correlation))
    # Find the time shift that maximizes correlation
    optimal_shift, max_correlation = max(cross_corr, key=lambda x: x[1])
    return optimal_shift

# Function to compute MSE between two columns with an optimal time shift
def compute_rmse_with_time_shift(df, col1, col2, max_shift):
    df_orig       = df.copy().dropna()
    mse_orig      = ((df_orig[col1] - df_orig[col2]) ** 2).mean()
    optimal_shift = find_optimal_time_shift(df, col1, col2, max_shift)
    # shift only column col1 in the dataframe
    shifted_col2 = df[col2].shift(optimal_shift)
    df_dropna    = pd.concat([df[col1], shifted_col2], axis=1).dropna()
    mse_shift    = ((df_dropna[col1] - df_dropna[col2]) ** 2).mean()
    return np.sqrt(mse_orig), np.sqrt(mse_shift)

def arrays(nt):

    # =================================================================
    # Radiation 
    SR    = np.zeros(shape=(nt,pucmparams.nR))   # W/m2 - net shortwave on roof
    SG    = np.zeros(shape=(nt,pucmparams.nG))   # W/m2 - net shortwave on ground
    SW    = np.zeros(shape=(nt,pucmparams.nW))   # W/m2 - net shortwave on wall
    ST    = np.zeros(nt)              # W/m2 - net shortwave on trees
    Sleaf = np.zeros(nt)              # W/m2 - net shortwave on leaves

    # =================================================================
    # Energy balance components
    QW = np.zeros(shape=(nt,pucmparams.nW))          # net heat flux at the walls [W/m2]
    QG = np.zeros(shape=(nt,pucmparams.nG))          # net heat flux at the ground [W/m2]
    QR = np.zeros(shape=(nt,pucmparams.nR))          # net heat flux at the roof [W/m2]
    LR = np.zeros(shape=(nt,pucmparams.nR))          # Longwave radiation at the roof
    LW = np.zeros(shape=(nt,pucmparams.nW))          # Longwave radiation at the wall
    LG = np.zeros(shape=(nt,pucmparams.nG))          # Longwave radiation at the ground
    HW = np.zeros(shape=(nt,pucmparams.nW))          # sensible heat flux at the walls
    HG = np.zeros(shape=(nt,pucmparams.nG))          # sensible heat flux at the ground
    HR = np.zeros(shape=(nt,pucmparams.nR))          # sensible heat flux at the roof
    LEC = np.zeros(nt)                    # Latent heat flux canyon
    LEG = np.zeros(shape=(nt,pucmparams.nG))         # Latent heat flux at the ground surface
    LER = np.zeros(shape=(nt,pucmparams.nR))         # Latent heat flux at the roof
    qR1 = np.zeros(shape=(nt,pucmparams.nR))         # Heat flux at the inner surface
    Lleaf  = np.zeros(nt)                           # Longwave radiation leaf
    LT     = np.zeros(nt)                           # Longwave tree
    Hcan   = np.zeros(nt)                           # Sensible heat flux above roof
    Hleaf  = np.zeros(nt)                           # Sensible heat flux at the leaf
    LEleaf = np.zeros(nt)                           # Latent heat flux from leaf [W/m2]
    RT  = np.zeros(nt)                    # Effective heat budget at the leaf
    RnW = np.zeros(shape=(nt,pucmparams.nW))         # Net available radiation on wall 
    RnG = np.zeros(shape=(nt,pucmparams.nG))         # Net available radiation on ground
    RnR = np.zeros(shape=(nt,pucmparams.nR))         # Net available radiation on roof
    ReW = np.zeros(nt)                    # Average across wall facets - net available radiation
    ReG = np.zeros(nt)                    # Average across ground facets - net available radiation
    ReR = np.zeros(nt)                    # Average across roof facets - net available radiation
    HWe = np.zeros(nt)                    # Average across wall facets - sensible heat flux
    HGe = np.zeros(nt)                    # Average across ground wall - sensible heat flux
    HRe = np.zeros(nt)                    # Average across roof facets - sensible heat flux
    LEGe = np.zeros(nt)                   # Average across ground facets - latent heat flux
    LERe = np.zeros(nt)                   # Average across roof facets - latent heat flux

    # =================================================================
    # Green's functions
    FoW = np.zeros(shape=(nt,pucmparams.nW)   )
    FoR = np.zeros(shape=(nt,pucmparams.nR)   )
    gW  = np.zeros(shape=(nt,2,pucmparams.nW) )
    gR  = np.zeros(shape=(nt,2,pucmparams.nR) )
    gG  = np.zeros(shape=(nt,pucmparams.nG)   )

    # =================================================================
    # Temperature 
    TW              = np.zeros(shape=(nt,pucmparams.nW))    # [K] T wall
    TG              = np.zeros(shape=(nt,pucmparams.nG))    # [K] T ground
    TR              = np.zeros(shape=(nt,pucmparams.nR))    # [K] T roof
    Tcan            = np.zeros(nt)                         # [K] T canyon
    TT              = np.zeros(nt)                         # [K] Leaf temperature 
    TWe             = np.zeros(nt)                         # Average temperature across wall facets
    TGe             = np.zeros(nt)                         # Average temperature across ground facets
    TRe             = np.zeros(nt)                         # Average temperature across roof facets
    TGrunoff        = np.zeros(shape=(nt,pucmparams.nG-1))  # Runoff temperature ground [K] 
    TGrunoffTop     = np.zeros(shape=(nt,pucmparams.nG-1))  # Runoff temperature ground [K] 
    TRrunoff        = np.zeros(shape=(nt,pucmparams.nR))    # Runoff temperature on the roof [K] 
    TRrunoffTop     = np.zeros(shape=(nt,pucmparams.nR))    # Runoff temperature on the roof [K] 
    Tdew            = np.zeros(nt)                         # Dew point temperature
    Twetbulb        = np.zeros(nt)                         # Wet bulb temperature

    # =================================================================
    # Hydrology terms
    WGv   = pucmparams.Ws * np.ones(shape=(nt,pucmparams.nL))            # volumetric water content profile at the ground (green portion)  [m3/m3]         
    WRv   = pucmparams.Ws * np.ones(nt)                       # Depth of water retention above roof (green roof)   
    delWR = pucmparams.dwR * np.ones(shape=(nt,max(1,pucmparams.nR-1)))  # Depth of water above roof (exclude green roof)
    delWG = pucmparams.dwG * np.ones(shape=(nt,pucmparams.nG-1))         # Depth of water above ground (exclude grass)
    WRi   = np.ones(shape=(nt,max(1,pucmparams.nR-1)))                  # Depth of water retention roof
    WGi   = np.ones(shape=(nt,pucmparams.nG-1))                         # Depth of water retention ground
    qcan  = np.zeros(nt)                                     # humidity inside canyon
    qW1   = np.zeros(shape=(nt,pucmparams.nW))                          # Heat flux at the inner surface (wall)
    SWG   = np.zeros(shape=(nt,pucmparams.nG))         # water budget at the ground [m_h2o/s]
    SWR   = np.zeros(shape=(nt,pucmparams.nR))         # water budget at the roof [m_h2o/s]
    DGe   = np.zeros(shape=(nt,pucmparams.nL))         # Hydraulic diffusivity [m2/s]
    KGe   = np.zeros(shape=(nt,pucmparams.nL))         # Hydraulic conductivity [m/s]
    Sroot          = np.zeros(shape=(nt,pucmparams.nL))      # Root water uptake [m/s]
    INFT           = np.zeros(shape=(nt,pucmparams.nL))      # infiltration [m/s]
    RoR            = np.zeros(shape=(nt,pucmparams.nR))      # roof surface runoff   [m/s]
    RoG            = np.zeros(shape=(nt,pucmparams.nG))      # ground surface runoff [m/s]
    Sq             = np.zeros(nt)
    waterR         = np.zeros(shape=(nt,pucmparams.nR))   # auxiliar to water available on the roof
    waterG         = np.zeros(shape=(nt,pucmparams.nG))   # auxiliar to water available on the ground
    qsG            = np.zeros(shape=(nt,pucmparams.nG))   # Saturated specific humidity on the ground [kg/kg]
    resGq          = np.zeros(shape=(nt,pucmparams.nG))   # Resistance
    GrunoffHeight  = np.zeros(shape=(nt,pucmparams.nG-1))
    GmeanRunVel    = np.zeros(shape=(nt,pucmparams.nG-1))
    RrunoffHeight  = np.zeros(shape=(nt,pucmparams.nR))
    RmeanRunVel    = np.zeros(shape=(nt,pucmparams.nR))
    volG           = np.zeros(shape=(nt,pucmparams.nG-1))
    qLG            = np.zeros(shape=(nt,pucmparams.nG-1))
    YG             = np.zeros(shape=(nt,pucmparams.nG-1))
    LeG            = np.zeros(shape=(nt,pucmparams.nG-1)) 
    volR           = np.zeros(shape=(nt,pucmparams.nR))
    qLR            = np.zeros(shape=(nt,pucmparams.nR))
    YR             = np.zeros(shape=(nt,pucmparams.nR))
    LeR            = np.zeros(shape=(nt,pucmparams.nR)) 
    QrunoffG       = np.zeros(shape=(nt,pucmparams.nG-1))
    QrunoffR       = np.zeros(shape=(nt,pucmparams.nR))

    # =================================================================
    # Atmospheric terms
    Ur     = np.zeros(nt)                        # Wind speed from log-profile
    Us     = np.zeros(nt)                        # Horizontal wind speed
    CEW    = np.zeros(shape=(nt,pucmparams.nW))
    CEG    = np.zeros(shape=(nt,pucmparams.nG))
    CER    = np.zeros(shape=(nt,pucmparams.nR))
    LobkG  = np.zeros(shape=(nt,pucmparams.nG))   # Obukhov lenght for fluxes from the ground (z0/L)
    LobkR  = np.zeros(shape=(nt,pucmparams.nR))   # Obukhov lenght for fluxes from the roof (z0/L)
    LobkC  = np.zeros(nt)                         # Obukhov lenght for fluxes from the canyon level (d/L)
    ustarG = np.zeros(shape=(nt,pucmparams.nG))   # Friction velocity [m/s]
    ustarR = np.zeros(shape=(nt,pucmparams.nR))   # Friction velocity [m/s]
    ustarC = np.zeros(nt)                         # Friction velocity [m/s]

    qsatG = np.zeros(shape=(nt,pucmparams.nG))
    qsatR = np.zeros(shape=(nt,pucmparams.nR))

    return (SR,SG,SW,ST,Sleaf,QW,QG,QR,LR,LW,LG,HW,HG,HR,LEC,LEG,LER,qR1,Lleaf,LT,
            Hcan,Hleaf,LEleaf,RT,RnW,RnG,RnR,ReW,ReG,ReR,HWe,HGe,HRe,LEGe,LERe,FoW,FoR,
            gW,gR,gG,TW,TG,TR,Tcan,TT,TWe,TGe,TRe,
            TGrunoff,TGrunoffTop,TRrunoff,TRrunoffTop,Tdew,Twetbulb,WGv,WRv,delWR,delWG,
            WRi,WGi,qcan,qW1,SWG,SWR,DGe,KGe,Sroot,INFT,RoR,RoG,Sq,
            waterR,waterG,qsG,resGq,GrunoffHeight,GmeanRunVel,RrunoffHeight,RmeanRunVel,
            Ur,Us,CEW,CEG,CER,LobkG,LobkR,LobkC,ustarG,ustarR,ustarC,qsatG,qsatR,
            volG, qLG, YG, LeG, volR, qLR, YR, LeR, QrunoffG, QrunoffR)
        