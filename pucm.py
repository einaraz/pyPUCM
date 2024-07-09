# import all functions from auxfunctions by name
from auxfunctions import *
from constants import Constants, pucmparams

thermal_exchange = True

def PUCM( Data, params, output='opt', verbose=False):
    print(" >>> started PUCM ")
    # Get data
    df = Data['df']
    ts = Data['ts']
    nt = Data['nt']
    dt = Data['dt']
    opt_var   = Data['opt_var']
    TurbFlux  = Data['TurbFlux']
    qmi_start = Data['qmi_start']

    # ====================================================
    
    SR,SG,SW,ST,Sleaf,QW,QG,QR,LR,LW,LG,HW,HG,HR,LEC,LEG,LER,qR1,Lleaf,LT,       \
    Hcan,Hleaf,LEleaf,RT,RnW,RnG,RnR,ReW,ReG,ReR,HWe,HGe,HRe,LEGe,LERe,FoW,FoR,  \
    gW,gR,gG,TW,TG,TR,Tcan,TT,TWe,TGe,TRe,                                       \
    TGrunoff,TGrunoffTop,TRrunoff,TRrunoffTop,Tdew,Twetbulb,WGv,WRv,delWR,delWG, \
    WRi,WGi,qcan,qW1,SWG,SWR,DGe,KGe,Sroot,INFT,RoR,RoG,Sq,                      \
    waterR,waterG,qsG,resGq,GrunoffHeight,GmeanRunVel,RrunoffHeight,RmeanRunVel, \
    Ur,Us,CEW,CEG,CER,LobkG,LobkR,LobkC,ustarG,ustarR,ustarC,qsatG,qsatR,        \
    volG, qLG, YG, LeG, volR, qLR, YR, LeR, QrunoffG, QrunoffR = arrays(nt)

    # ====================================================
    # Compute solar radiation on different surfaces
    for i in range(nt):
        SR[i], SG[i], SW[i], ST[i], Sleaf[i] = NetShortwaveRadiation( df['Sd'].iloc[i], df['qzenith'].iloc[i], df['qazimuth'].iloc[i], i)

    # ====================================================
    # compute Green's functions: 
    #      will be used later to solve the heat equation
    n   = 20
    
    # For wall
    for i in range(pucmparams.nW):
        FoW[:,i]  = pucmparams.alW[i] * ts /(pucmparams.dW**2)
        gW[:,:,i] = Green( FoW[:,i], pucmparams.dW, pucmparams.kW[i], pucmparams.alW[i], ts, n)

    # For roof
    for i in range(pucmparams.nR):
        FoR[:,i]  = pucmparams.alR[i] * ts /(pucmparams.dR**2)
        gR[:,:,i] = Green( FoR[:,i], pucmparams.dR, pucmparams.kR[i], pucmparams.alR[i], ts, n)

    # Ground
    for i in range(pucmparams.nG):
        gG[:,i] = 2.0 * np.sqrt( pucmparams.alG[i] * ts / np.pi )/ pucmparams.kG[i]   # for road
    
    # ====================================================
    # Initial temperature and soil moisture
    #    Set initial values

    TW[0,:]  = pucmparams.Tsi + Constants.KK # wall temperature [K]
    TR[0,:]  = pucmparams.Tsi + Constants.KK # roof  temperature [K]
    Tcan[0]  = df['Ta'].iloc[0]             # canyon  temperature [K]
    qcan[0]  = df['qa'].iloc[0]             # specific humidity [kg_wv/kg_air]
    TT[0]    = df['Ta'].iloc[0]             # Tree temperature

    for gi in range(pucmparams.nG):
        TG[0,gi]  = pucmparams.TGi[gi] + Constants.KK 
    
    TGrunoff[0,:]    = df['Ta'].iloc[0]
    TGrunoffTop[0,:] = df['Ta'].iloc[0]

    WGv[0,:] = pucmparams.smc_profile   # Young-Hee
    WRv[0]   = pucmparams.qmi

    WRi[0,:] = pucmparams.poR * ( dt * df['Pd'].iloc[0] )/pucmparams.dwR
    WGi[0,:] = ( dt * df['Pd'].iloc[0]  )/pucmparams.dwG


    # ====================================================
    # Time integration
    niter         = 0
    Maxi          = 200
    tol           = 1.0E-3

    for i in range(nt):   
        if verbose:  
            print(" time: ", df.index[i-1], TG[i-1,0], GrunoffHeight[i-1,0])
        nit0 = 0             # start number iterations at 0
        ok   = 0             # monitering iteration
    
        
        # Compute Obukhov length -------------------------------------------
        
        if i > 0:
            # Ground
            for j in range(pucmparams.nG):
                pt1        = ( Constants.kappa * Constants.g / Tcan[i-1]) 
                pt2        = ( HG[i-1,j]/Constants.Cpd/df['ra'].iloc[i-1] + 0.61 * Tcan[i-1] * LEG[i-1,j]/Constants.Lv/df['ra'].iloc[i-1])
                if abs(pt1 * pt2) > 0:
                    LobkG[i,j] = - ustarG[i-1,j]**3.0 / (pt1 * pt2)
                else: LobkG[i,j]
            # Roof
            for j in range(pucmparams.nR):
                pt1        = (Constants.kappa * Constants.g / df['Ta'].iloc[i-1]) 
                pt2        = ( HR[i-1,j]/Constants.Cpd/df['ra'].iloc[i-1] + 0.61 * df['Ta'].iloc[i-1] * LER[i-1,j]/Constants.Lv/df['ra'].iloc[i-1])
                if abs(pt1 * pt2) > 0:
                    LobkR[i,j] = - ustarR[i-1,j]**3.0 / (pt1 * pt2)
                else: LobkR[i,j] = 0
            # Canyon
            pt1        = (Constants.kappa * Constants.g / Tcan[i-1])
            pt2        = ( Hcan[i-1]/Constants.Cpd/df['ra'].iloc[i-1] + 0.61 * Tcan[i-1] * LEC[i-1]/Constants.Lv/df['ra'].iloc[i-1])
            if abs(pt1 * pt2) > 0:
                LobkC[i]   = - ustarC[i-1]**3.0 / (pt1 * pt2)
            else: LobkC[i] = 0
            
            # Water budget ===============================================================================
            
            # temporal variation of water layer at the ground [m/s]: rain - evaporation - runoff
            for j1 in range(pucmparams.nG):
                SWG[i,j1] = df['Pd'].iloc[i] - LEG[i-1,j1]/Constants.Lv/Constants.rW - RoG[i-1,j1]
                
            # temporal variation of  water at the roof [m/s]: rain - evaporation - runoff
            for j1 in range(pucmparams.nR):
                SWR[i,j1] = df['Pd'].iloc[i] - LER[i-1,j1]/Constants.Lv/Constants.rW - RoR[i-1,j1]
                
            # Runoff dynamics ===============================================================================    
            for j in range(pucmparams.nG-1):
                GrunoffHeight[i,j] = GrunoffHeight[i-1,j] + SWG[i,j] * dt
                if GrunoffHeight[i,j] < 0:
                    GrunoffHeight[i,j] = 0.0
                # accumulate rain in the previous 24 hours taking dt into account
                if (SWG[i,j] <= 0) and (GrunoffHeight[i,j] <= pucmparams.G_hc0[j]) and (rain_06h_accumulated<=0):
                    GrunoffHeight[i,j] = 0.0
                
                if GrunoffHeight[i,j] > pucmparams.G_StLayer[j]:
                    runoffVelocity = (Constants.g * pucmparams.G_slope[j] / (3 * Constants.vi) ) * (GrunoffHeight[i,j] - pucmparams.G_StLayer[j])**2.0  
                else:
                    runoffVelocity = 0
                    
                GmeanRunVel[i,j] = runoffVelocity                                             
                RoG[i,j]         = GmeanRunVel[i,j] * ((GrunoffHeight[i,j]-pucmparams.G_StLayer[j]) / pucmparams.G_surf_length[j])   # runoff from asphalt [m/s]
                
            for j in range(pucmparams.nR):
                RrunoffHeight[i,j] = RrunoffHeight[i-1,j] + SWR[i,j] * dt
            
                if RrunoffHeight[i,j] < 0:
                    RrunoffHeight[i,j] = 0.0
                if (SWR[i,j] <= 0) and (RrunoffHeight[i,j] <= pucmparams.R_hc0[j]) and (rain_06h_accumulated<=0):
                    RrunoffHeight[i,j] = 0.0
                
                if RrunoffHeight[i,j] > pucmparams.R_StLayer[j]:
                    runoffVelocity = (Constants.g * pucmparams.R_slope[j] / (3 * Constants.vi) ) * (RrunoffHeight[i,j] - pucmparams.R_StLayer[j])**2.0  
                else:
                    runoffVelocity = 0
                    
                RmeanRunVel[i,j] = runoffVelocity                                             
                RoR[i,j]         = RmeanRunVel[i,j] * ((RrunoffHeight[i,j]-pucmparams.R_StLayer[j]) / pucmparams.R_surf_length[j])   # runoff from asphalt [m/s]
                
        else:
            LobkG[i,:] = 0
            LobkR[i,:] = 0
            LobkC[i]   = 0
            SWG[i,:] = 0 
            SWR[i,:] = 0 
            GrunoffHeight[i,:] = 0 
            GmeanRunVel[i,:] = 0 
            RoG[i,:] = 0 
            RmeanRunVel[i,:] = 0 
            RoR[i,:] = 0
            
        # =============================================================
        # Wind dynamics inside canyon
        
        # wind speed inside the canyon from log-profile (town roughness, Z0) above the canyon [m/s]
        Ur[i] = 2.0 * df['Ua'].iloc[i] * np.log(pucmparams.Zr/3.0/pucmparams.Z0)/np.log((pucmparams.Za-pucmparams.Zr+pucmparams.Zr/3.0)/pucmparams.Z0)/np.pi
        # Horizontal wind speed (below the canyon) [m/s]
        Us[i] = Ur[i] * np.exp(-0.25 * pucmparams.h / pucmparams.w)
        # Accumulated rainfall in the previous 24 hours [m]
        rain_06h_accumulated = df['Pd'][(df.index[i]-timedelta(hours=6)):(df.index[i])].sum() * dt

        # Compute heat exchange coefficients based on the type of method used for turbulent fluxes
        if TurbFlux == "MOST":
            pass
        elif TurbFlux == "SLUCM":
            RW = (6.15 + 4.18 * Us[i])
            if Us[i] > 5: RW = 7.51 * Us[i]**0.78
            CEW[i,:] = RW / ( df['ra'].iloc[i] * Constants.Cpd * Us[i] )
            CEG[i,:] = RW / ( df['ra'].iloc[i] * Constants.Cpd * Us[i] )
        elif TurbFlux == 'PUCM':
            RW     = ( 11.8 + 4.2 * Us[i])
            CEW[i,:] = RW / ( df['ra'].iloc[i] * Constants.Cpd * Us[i] )
    
        # =============================================================
        # Aproximate rain temperature
        Ta                      = df['Ta'].iloc[i] - 273.15
        Twetbulb[i], Tdew[i]    = TwetbulbTdew( Ta, df['RH'].iloc[i], Constants ) 
        Train                   = Twetbulb[i]
        
        while( (ok==0) & (nit0 < Maxi) ):
            nit0 = nit0 + 1
    
            # =============================================================
            # quantities to check for convergence
            x0 = SWG[i,0]
            x1 = qW1[i,0] 
            x2 = TR[i,0]
            x3 = WGv[i,0] 
            x4 = Tcan[i] 
            x5 = TW[i,0] 
            x6 = TG[i,0] 
            x7 = TG[i,1]
    
            # =============================================================
            # Exterior boundary condition to solve heat flux equation [W/m2] 
            #  Term f1 in 38
    
            # Initial guess for surface energy budget terms
            if nit0 == 0:
                LW[i,:]  = LW[i-1,:]
                HW[i,:]  = HW[i-1,:]
                LR[i,:]  = LR[i-1,:]
                HR[i,:]  = HR[i-1,:]
                LER[i,:] = LER[i-1,:]
                LG[i,:]  = LG[i-1,:]
                HG[i,:]  = HG[i-1,:]
                LEG[i,:] = LEG[i-1,:]
     
            QW[i,:]  = SW[i,:]  + LW[i,:]  - HW[i,:]
            QR[i,:]  = SR[i,:]  + LR[i,:]  - HR[i,:] - LER[i,:]
            QG[i,:]  = SG[i,:]  + LG[i,:]  - HG[i,:] - LEG[i,:]
            
            # Energy exchange term between ground and runoff
            if thermal_exchange:
                if df['Pd'].iloc[i] > 0:
                    P        = df['Pd'].iloc[i] * dt
                    for j in range(pucmparams.nG-1):
                        delta    = np.sqrt( 4 * pucmparams.alG[j] * dt )
                        beta     = delta * (  pucmparams.cG[j] ) /( 2 * P * (1000 * Constants.ch2o) )
                        XZ       = beta / (1 + beta)
                        Hro      = - df['Pd'].iloc[i] * (1000 * Constants.ch2o) * (TG[i-1,j] - Train) * XZ
                        QG[i,j]  = QG[i,j] + Hro
                        QrunoffG[i,j] = Hro
                    for j in range(pucmparams.nR):
                        delta    = np.sqrt( 4 * pucmparams.alR[j] * dt )
                        beta     = delta * (  pucmparams.cR[j] ) /( 2 * P * (1000 * Constants.ch2o) )
                        XZ       = beta / (1 + beta)
                        Hro      = - df['Pd'].iloc[i] * (1000 * Constants.ch2o) * (TR[i-1,j] - Train) * XZ
                        QR[i,j]  = QR[i,j] + Hro
                        QrunoffR[i,j] = Hro
            else:
                QrunoffR[i,:] = 0
                QrunoffG[i,:] = 0
                
            # =============================================================
            # Computing temperatures
            if i > 0:
                # Temperature of walls facets
                for j in range(pucmparams.nW):
                    TW[i,j], qW1[i,j] = TGF( gW[:,:,j], QW[:,j], qW1[:,j], i)
                TW[i,:] = TW[i,:] + TW[0,:]
                
                # temperature at the roof
                for j in range(pucmparams.nR):
                    TR[i,j], qR1[i,j] = TGF( gR[:,:,j], QR[:,j], qR1[:,j], i)
                TR[i,:] = TR[i,:] + TR[0,:]
    
                # Temperature of ground facets
                for j in range(pucmparams.nG):
                    TG[i,j] = TG[0,j] + 0.5 * gG[1,j] * QG[i,j] + np.trapz( x=gG[:i,j], y=np.concatenate(( [0], QG[:(i-1),j][::-1] )) )
                    
                # ==================================================================
                # Vegetated/bare soil option
                # Hydraulic conductivity and diffusivity - Corsby-Chen(LSM) model 
                DGe[i,:], KGe[i,:] = DKeff( WGv[i,:])
    
                # ==================================================================
                # Water content availability [%]
                # Ground (ignores soil portion, which is last in the array)
                # Modify available water at the surface for asphalt based on runoff height     
                WGi[i,:]           = GrunoffHeight[i,:]/pucmparams.G_StLayer[:]
                # Roof (ignores last option, which is green roof)
                WRi[i,:]           = RrunoffHeight[i,:]/pucmparams.R_StLayer[:]
                # Green roof water retention (last option in SWR)
                WRv[i]             = WRv[i-1]   + dt * SWR[i,-1]/pucmparams.dvR
                    
                # ==================================================================
                # Compute soil water content, stress index for vegetated ground
                
                for j2 in range(pucmparams.nL):
                    # normalized soil moisture
                    normq =  ( WGv[i,j2] - pucmparams.Wr ) / (pucmparams.Ws - pucmparams.Wr)
                    if ( normq == 1 ):
                        alphai = 1.0 # stress index
                    elif ( (normq >= pucmparams.qc1) & (normq < 1.) ):
                        alphai = 1.0 # stress index
                    else:
                        alphai = normq/pucmparams.qc1 # stress index
    
                # Weighted stress index
                ind_stress = np.sum( pucmparams.rootl * alphai)  
                
                # total water uptake by roots 
                for j2 in range(pucmparams.nL):
                    Sroot[i,j2] = 2 * pucmparams.ft * pucmparams.Aleaf * LEleaf[i]/ (pucmparams.fG[-1] * pucmparams.W) / Constants.Lv / Constants.rW * pucmparams.rootl[j2] * alphai/ind_stress
                        
                # Solves 1D Richardson's equation
                # WGv: water content (%), INFT: infiltration [m/s]
                WGv[i,:] , INFT[i,:] = SoilInfiltrationRootUptake( SWG[i,-1], 0, WGv[i-1,:], DGe[i,:], KGe[i,:], dt, Sroot[i,:] )
    
            # ==================================================================
            # Force minimun and maximum soil water contents to be equal to residual and water capacity 
            WGv[i,:] = np.maximum( pucmparams.Wr * np.ones( len(WGv[i,:]) ), WGv[i,:])
            WGv[i,:] = np.minimum( pucmparams.Ws * np.ones( len(WGv[i,:]) ), WGv[i,:]) 
            # Forces water content at the ground (asphalt and concrete) between 0 and 1 
            WGi[i,:] = np.maximum(  0 * np.ones( len(WGi[i,:]) ), WGi[i,:]) 
            WGi[i,:] = np.minimum(  1 * np.ones( len(WGi[i,:]) ), WGi[i,:]) 
            # Forces water content at the green roof to be between Wr and Ws
            WRv[i]   = np.maximum( pucmparams.Wr, WRv[i]  )
            WRv[i]   = np.minimum( pucmparams.Ws, WRv[i]  )
            # Forces water content at the roof between 0 and roof porosity
            WRi[i,:] = np.maximum(  0 * np.ones( len(WRi[i,:]) ), WRi[i,:] )
            WRi[i,:] = np.minimum(  1 * np.ones( len(WRi[i,:]) ), WRi[i,:] )       
                
            # using Budykho linear method to determine beG
            #w_critical = pucmparams.Wr 
            #w_sat      = pucmparams.Ws 
            #if WGv[i,0] > 0.47:
            #    beG = 1.0
            #else:
            #    beG = ( WGv[i,0] - w_critical ) / ( w_sat - w_critical )
                
            # ==================================================================
            # Update temperatures for energy budget (average of facets)
            T1     = TW[i,:]   # wall temperature [K]
            T2     = TG[i,:]   # ground temperature [K]
            T3     = TR[i,:]   # roof temperature [K]
            TWe[i] = np.dot( pucmparams.fW, np.squeeze(T1) ) # fW is the fraction of the different types of wall
            TGe[i] = np.dot( pucmparams.fG, np.squeeze(T2) ) # fG is the fraction of the different types of ground
            TRe[i] = np.dot( pucmparams.fR, np.squeeze(T3) ) # fR is the fraction of the different types of roof
                
            # Compute net longwave radiation at all facets [W/m2]
            LR[i,:], LG[i,:], LW[i,:], LT[i], Lleaf[i] = NetLongwaveRadiation( df['Ld'].iloc[i], T3, T2, T1, TT[i], TGe[i], TWe[i])
                
            # ==================================================================
            # Update turbulent quantities on the roof (u*, T*, q*)
            for j in range(pucmparams.nR):
                T_temp       = TR[i,j]            
                qsatR[i,j]   = qsat(T_temp, df['Pa'].iloc[i])
                
                if TurbFlux == "MOST":
                    LER[i,j], HR[i,j], CER[i,j], ustarR[i,j] =  TurbulentFluxesMOST( pucmparams.Za-pucmparams.d, pucmparams.ZmR[j], pucmparams.ZhR[j], df['Ua'].iloc[i], 0, df['Ta'].iloc[i], T_temp, df['qa'].iloc[i], qsatR[i,j], LobkR[i,j], df['ra'].iloc[i])
                if TurbFlux in ['SLUCM', 'PUCM']:
                    LER[i,j], HR[i,j], CER[i,j], ustarR[i,j] = TurbulentFluxesRoofGroundAero( df['Ta'].iloc[i], df['qa'].iloc[i], qsatR[i,j], df['ra'].iloc[i], T_temp, df['Ua'].iloc[i], pucmparams.Za-pucmparams.Zr, pucmparams.ZmR[j], pucmparams.ZhR[j] )
                
                if RrunoffHeight[i,j] > 0.0: #pucmparams.R_hc0[j]:
                    waterflux = LER[i,j] / Constants.Lv
                    if (waterflux/Constants.rW) * dt > RrunoffHeight[i,j]:
                        LER[i,j] = Constants.Lv * RrunoffHeight[i,j] * Constants.rW /dt
                else:
                    LER[i,j] = 0
                    
            # Effective evaporation: in this multiplication, WRi works as a reduction factor 
            LER[i,:] = WRi[i,:] * LER[i,:]
            
            # ==================================================================
            # update turbulent quantities at the canyon level 
            if TurbFlux == "MOST":
                LEC[i], Hcan[i], _, ustarC[i] =  TurbulentFluxesMOST( pucmparams.Za, pucmparams.d, pucmparams.d, df['Ua'].iloc[i], Us[i], df['Ta'].iloc[i], Tcan[i], df['qa'].iloc[i], qcan[i], LobkC[i], df['ra'].iloc[i])
            if TurbFlux in ['SLUCM', 'PUCM']:
                LEC[i], Hcan[i], ustarC[i] = TurbulentFluxesCanyonAero( df['Ta'].iloc[i], df['qa'].iloc[i], df['ra'].iloc[i], Tcan[i], qcan[i], df['Ua'].iloc[i], pucmparams.Za-pucmparams.d, pucmparams.Zmc, pucmparams.Zhc)

            # Get transfer coefficients from canopy: used later to find Tcan and qcan
            if abs(Tcan[i] - df['Ta'].iloc[i]) > 0:
                flux_res_can_H = ( Hcan[i] / (Constants.Cpd * df['ra'].iloc[i]) ) / (Tcan[i]-df['Ta'].iloc[i])
            else:
                flux_res_can_H = 0

            if abs(qcan[i] - df['qa'].iloc[i]) > 0:
                flux_res_can_L = ( LEC[i] / (Constants.Lv * df['ra'].iloc[i]) ) / (qcan[i] - df['qa'].iloc[i])
            else:
                flux_res_can_L = 0

            # ==================================================================
            # Sensible heat flux from the different types of walls
            for j in range(pucmparams.nW):
                if TurbFlux == "MOST":
                    HW[i,j]  =  TurbulentFluxesMOST( pucmparams.W/2, pucmparams.ZmW[j], pucmparams.ZhW[j], Us[i], 0, Tcan[i], TW[i,j], qcan[i], 0, 0, df['ra'].iloc[i], 'wall')
                if TurbFlux in ['SLUCM', 'PUCM']:
                    HW[i,j] = TurbulentFluxesWallSLUCM( df['ra'].iloc[i], CEW[i,j], Us[i], TW[i,j], Tcan[i])
            
            # Get transfer coefficients from wall: used later to find Tcan and qcan
            if max(abs(TW[i,:] - Tcan[i])) > 0:
                flux_res_wall_H = ( HW[i,:] / (Constants.Cpd * df['ra'].iloc[i]) ) / (TW[i,:] - Tcan[i])
            else:
                flux_res_wall_H = 0

            # ==================================================================
            # Update turbulent quantities at the ground level
                        
            # Loop over ground facets
            for j in range(pucmparams.nG):
                T_temp       = TG[i,j]
                qsatG[i,j]   = qsat(T_temp, df['Pa'].iloc[i])
                
                if TurbFlux == "MOST":
                    LEG[i,j], HG[i,j], CEG[i,j], ustarG[i,j] = TurbulentFluxesMOST( pucmparams.d, pucmparams.ZmG[j], pucmparams.ZhG[j], Us[i], 0, Tcan[i], T_temp, qcan[i], qsatG[i,j], LobkG[i,j], df['ra'].iloc[i])
                if TurbFlux == "SLUCM":
                    LEG[i,j], HG[i,j], ustarG[i,j]           = TurbulentFluxesGroundSLUCM( df['ra'].iloc[i], CEG[i,j], Us[i], T_temp, Tcan[i], qcan[i], qsatG[i,j])
                if TurbFlux == "PUCM":
                    LEG[i,j], HG[i,j], CEG[i,j], ustarG[i,j] = TurbulentFluxesRoofGroundAero( Tcan[i], qcan[i], qsatG[i,j], df['ra'].iloc[i], T_temp, Us[i], pucmparams.d, pucmparams.ZmG[j], pucmparams.ZhG[j] )
                
                resGq[i,j] = 1.0 / (CEG[i,j] * Us[i])
                
                if ( j <= 1 ):
                    waterflux = LEG[i,j]/Constants.Lv
                    if GrunoffHeight[i,j] > 0.0: #pucmparams.G_hc0[j]:
                        if (waterflux/Constants.rW)*dt > GrunoffHeight[i,j]:
                            LEG[i,j] = Constants.Lv * GrunoffHeight[i,j] * Constants.rW /dt
                    else:
                        LEG[i,j] = 0
    
            # Effective latent heat from ground
            LEG[i,0:(pucmparams.nG-1)]      = WGi[i,:] * LEG[i,0:(pucmparams.nG-1)]
            # Latent heat flux for vegetation 
                    
            RsG                            = StomatalResistance(40., SG[i,pucmparams.nG-1], pucmparams.LAIgrass, WGv[i,0], pucmparams.Wr, pucmparams.Ws, Tcan[i], qcan[i], qsatG[i,pucmparams.nG-1], df['Pa'].iloc[i] )
            LEG[i,pucmparams.nG-1]         = df['ra'].iloc[i] * Constants.Lv * ( qsatG[i,pucmparams.nG-1] - qcan[i] )/( resGq[i,pucmparams.nG-1] + RsG )

            # Get transfer coefficients from ground: used later to find Tcan and qcan
            if max(abs(TG[i,:] - Tcan[i])) > 0:
                flux_res_ground_H = ( HG[i,:] / (Constants.Cpd * df['ra'].iloc[i]) ) / (TG[i,:] - Tcan[i])
            else:
                flux_res_ground_H = 0
            
            if max(abs(qsatG[i,:] - qcan[i])) > 0:
                flux_res_ground_L = ( LEG[i,:] / (Constants.Lv * df['ra'].iloc[i]) ) / (qsatG[i,:] - qcan[i])
            else:
                flux_res_ground_L = 0

            # ==================================================================
            # Tree fluxes
                
            # Leaf boundary-layer resistance
            Rbl       = LeafBLMresistance( Us[i], pucmparams.Aleaf, pucmparams.aleaf )  
            Hleaf[i]  = df['ra'].iloc[i] * Constants.Cpd * ( TT[i] - Tcan[i] )/(1.274 * Rbl)   # modified by the ratio of molecular diffusion for heat and moisture
            qsT       = qsat( TT[i], df['Pa'].iloc[i] )
    
            # Leaf stomatal resistance
            RsT       = StomatalResistanceTree( 100., Sleaf[i], WGv[i, 0:5], pucmparams.Wr, pucmparams.Ws, Tcan[i], qcan[i], qsT, pucmparams.rootl[0:5], pucmparams.LAItree )
            LEleaf[i] = LE_leaf( Sleaf[i] + Lleaf[i], Rbl, RsT, df['ra'].iloc[i], Tcan[i], qcan[i], qsT)
            
            # Get transfer coefficients from canopy: used later to find Tcan and qcan
            if abs(TT[i] - Tcan[i]) > 0:
                flux_res_leaf_H = ( Hleaf[i] / (Constants.Cpd * df['ra'].iloc[i]) ) / (TT[i] - Tcan[i])
            else:
                flux_res_leaf_H = 0

            # ==================================================================
            # compute effective heat budgets (net available energy)
            
            RT[i]    = Lleaf[i] + Sleaf[i]
            RnR[i,:] = LR[i,:]  + SR[i,:]
            RnW[i,:] = LW[i,:]  + SW[i,:]
            RnG[i,:] = LG[i,:]  + SG[i,:]
    
            # Average each surface based on the fractions of different facets
            ReW[i]   = np.dot(RnW[i,:], pucmparams.fW)  # R wall
            ReG[i]   = np.dot(RnG[i,:], pucmparams.fG)  # R ground
            ReR[i]   = np.dot(RnR[i,:], pucmparams.fR)  # R roof
            HRe[i]   = np.dot(HR[i,:] , pucmparams.fR)  # H roof
            HWe[i]   = np.dot(HW[i,:] , pucmparams.fW)  # H wall
            HGe[i]   = np.dot(HG[i,:] , pucmparams.fG)  # H ground
            LERe[i]  = np.dot(LER[i,:], pucmparams.fR)  # LE roof
            LEGe[i]  = np.dot(LEG[i,:], pucmparams.fG)  # LE ground
    
            # ==================================================================
            # Canopy temperature and humidity 
            
            # Young-Hee Ryu
            # Compute canopy temperature diagnostically
            hcanTcan       = flux_res_can_H
            hcan           = flux_res_can_H * df['Ta'].iloc[i]
            hwallsTcan     = (2*pucmparams.h/pucmparams.w) * np.sum(pucmparams.fW * flux_res_wall_H)
            hwallsTwall    = (2*pucmparams.h/pucmparams.w) * np.sum(pucmparams.fW * flux_res_wall_H * TW[i,:])
            hgroundTcan    =  np.sum( pucmparams.fG * flux_res_ground_H )
            hgroundTground =  np.sum( pucmparams.fG * flux_res_ground_H * TG[i,:] )
            hleafT         = 2 * pucmparams.ft * pucmparams.Aleaf * flux_res_leaf_H * TT[i] / pucmparams.W
            hleafTcan      = 2 * pucmparams.ft * pucmparams.Aleaf * flux_res_leaf_H / pucmparams.W
            Tcan[i]        = ( hcan + hwallsTwall + hgroundTground + hleafT) / ( hcanTcan + hwallsTcan + hgroundTcan + hleafTcan )
                
            # Compute canopy humidity diagnostically
            lgroundTcan    = np.sum( pucmparams.fG * flux_res_ground_L )
            lgroundTground = np.sum( pucmparams.fG * flux_res_ground_L * qsatG[i,:] )
            lcan           = flux_res_can_L * df['qa'].iloc[i]
            lcanTcan       = flux_res_can_L 
            ltree          = ((pucmparams.ft * 2.0 * pucmparams.Aleaf / pucmparams.W) * LEleaf[i])/(Constants.Lv * df['ra'].iloc[i])
            qcan[i]        = ( lcan + lgroundTground + ltree ) / ( lcanTcan + lgroundTcan  )
            
            if ( i < (nt-1) ):
                # Tree temperature uses canopy temperature from previous time step only: it should be stable
                TT[i+1]        = TT[i] + dt/(640.0) * ( Sleaf[i] + Lleaf[i] - Hleaf[i] - LEleaf[i] ) # 640 J/m2/K is the heat capacity per leaf

            # ==================================================================
            # error : 
            if abs(SWG[i,0]) > 0: 
                errorSWG = abs( x0 / SWG[i,0] - 1)
            else: 
                errorSWG = 0

            if qW1[i,0] > 0: 
                errorqW1 = abs( x1 / qW1[i,0] - 1)
            else: 
                errorqW1 = 0
            
            #err = [ errorSWG, 
            err = [ errorqW1, abs( x2 / TR[i,0] - 1), abs( x3 / WGv[i,0] - 1 ),
                    abs( x4 / Tcan[i] -1),   abs( x5 / TW[i,0] - 1), abs( x6 /  TG[i,0] -1  ), abs( x7 / TG[i,1] -1 ) ]
            emax = np.nanmax(err)
            
            erros_dict = {
                'swg':  errorSWG, 
                'qw1':  errorqW1, 
                'TR0':   abs( x2 / TR[i,0] - 1), 
                'WGv0':  abs( x3 / WGv[i,0] - 1 ),
                "Tcan": abs( x4 / Tcan[i] -1),   
                'TW0':   abs( x5 / TW[i,0] - 1), 
                'TG0':   abs( x6 /  TG[i,0] -1 ), 
                'TG1':   abs( x7 / TG[i,1] -1 )
                          }
                        
            if ( emax < tol):
                ok=1
    
        if nit0 >= Maxi:
            print(erros_dict)
            raise ValueError('maximum no of iteration exceeded.')
        niter = niter + nit0
    
    # ==================================================================
    # Final temperature in Celsius
    TW   =   TW - Constants.KK
    TG   =   TG - Constants.KK
    TR   =   TR - Constants.KK
    Tcan = Tcan - Constants.KK
    TT   =   TT - Constants.KK
    TWe  =  TWe - Constants.KK
    TGe  =  TGe - Constants.KK
    TRe  =  TRe - Constants.KK
    TGrunoff    = TGrunoff    - Constants.KK
    TGrunoffTop = TGrunoffTop - Constants.KK
    TRrunoff    = TRrunoff    - Constants.KK
    TRrunoffTop = TRrunoffTop - Constants.KK
    Tdew        = Tdew        - Constants.KK
    Twetbulb    = Twetbulb    - Constants.KK
    
    # ==================================================================
    # Average heat fluxes from roofs (r) and from canyons (w), which include wall and roads
    Hu     = pucmparams.r * np.squeeze(HRe)  + pucmparams.w * np.squeeze(Hcan)
    LEu    = pucmparams.r * LERe             + pucmparams.w * LEC
    WGn    = np.squeeze(WGv[:,0])           # model soil moisture, [%]
    Wroof  = pucmparams.r / pucmparams.w * pucmparams.W
    Rnet   = pucmparams.r * ReR + pucmparams.w * ReG + 2.0 * pucmparams.h * ReW + 2.0 * pucmparams.ft * (2.0 * np.pi * pucmparams.aleaf) * pucmparams.LAItree / np.pi /( pucmparams.W + Wroof ) * RT
    qcan   = qcan * 1000.
    
    # Saving results -------------------------------------------------------
    results = {}
    for aux,an in [  [Tcan, 'Tcan'],         # Canyon temperature
                     [TR, 'TR1'],             # Roof temperature
                     [TW, 'TW'],             # Wall temperature
                     [TG, 'TG'],             # Ground temperature 
                     [Twetbulb, 'Twetbulb'], # Wet bulb temperature
                     [Tdew, 'Tdew'],         # Dew point temperature
                     [Rnet, 'Rnet'],     # Net solar radiation over canyon
                     [Hu, 'Hu'],         # average heat flux from roofs and roads
                     [LEu, 'LEu'],       # Average sensible heat flux from roofs and roads
                     [HR, 'HR'],         # Roof sensible heat flux
                     [HW, 'HW'],         # Wall sensible heat flux
                     [HG, 'HG'],         # Ground sensible heat flux
                     [ Hleaf, 'Htree'],  # Leaf sensible heat flux
                     [LER, 'LER'],       # Roof latent heat flux
                     [LEG, 'LEG'],       # Ground latent heat flux
                     [LEleaf, 'LEtree'], # Leaf latent heat flux
                     [SR, 'SR'],         # Roof net solar radiation
                     [LR, 'LR'],         # Roof net longwave radiation
                     [SW, 'SW'],         # Wall net solar radiation
                     [LW, 'LW'],         # Wall longwave radiation
                     [SG, 'SG'],         # Ground net solar radiation
                     [LG, 'LG'],         # Ground longwave radiation
                     [ST, 'ST'],         # Tree net solar radiation
                     [LT, 'LT'],         # Tree longwave radiation
                     [qcan, 'qcan'],     # Canyon specific humidity
                     [WGn, 'WGn'],       # Water volume content in the ground
                     [TT, 'TT'],         # Tree temperature
                     [Hcan, 'Hcan'],     # Canyon sensible heat flux
                     [LEC, 'LEC'],       # Canyon latent heat flux
                     [ QG, 'QG'],
                     [ QW, 'QW'],
                     [ QR, 'QR'],
                     [ WGi[:,0], 'WGi'], 
                     [ WGi[:,1], 'WGi1'], 
                     [ WRi[:,0], 'WRi'], 
                     [ SWG[:,0], 'SWG' ],
                     [ WGv[:,0], 'WGv1'],
                     [ WGv[:,-1], 'WGv2'],
                     [ QrunoffG, 'QrunoffG'],
                     [ QrunoffR, 'QrunoffR'],
                     [GrunoffHeight, 'GrunoffHeight'],    # runoff height [m] 
                     [ RoG[:,0], 'RoG'],
                     [GmeanRunVel, 'GmeanRunVel'],
                     [TGrunoff,    'TGrunoff'],
                     [TGrunoffTop, 'TGrunoffTop'],
                     [RrunoffHeight, 'RrunoffHeight'],    # runoff height [m]
                     [ RoR[:,0], 'RoR'],
                     [RmeanRunVel, 'RmeanRunVel'],
                     [TRrunoff,    'TRrunoff'],
                     [TRrunoffTop, 'TRrunoffTop'],
                     [ LobkC, 'LobkC'] ,
                     [ LobkG, 'LobkG'] ,
                     [ LobkR, 'LobkR'] ,
                     [ ustarG, 'ustarG'],
                     [ ustarR, 'ustarR'],
                     [ ustarC, 'ustarC'],
                     ]:      # mean velocity of runoff [m/s]
    
        s   = aux.shape
        key = an
    
        if len(s) > 1: 
            s = s[1]
            for ii in range(s):
                if s > 1:
                    key = "%s%d"%(an,ii+1)
                results[key] = np.array(aux[:,ii])
        else:
            key          = an
            results[key] = np.array(aux[:])
    
    results = pd.DataFrame( results, index=df.index[:-1] )
    if output == 'opt':
        # return the dataframe but average every 1 minute but still containing datatimes every dt seconds
        results = results.resample('1min').mean()
        results = results.resample('%ss'%dt).mean()
        return results[[opt_var,'Rnet']].copy()
    else:
        return results