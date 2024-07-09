import numpy as np

class Constants:
    KK    = 273.15         # Celsius-Kelvin conversion
    Rd    = 287            # gas constant for dry air [J/kg/K]
    Rv    = 461.5          # gas constant for vapor
    rW    = 1e3            # density of water [kg/m3]
    Cpd   = 1005           # heat capacity of dry air [J/kg/K]
    Lv    = 2.26e6         # latent heat of vaporization [J/kg]
    ss    = 5.67e-8        # Stephan-Boltzmann constant [J/s/m2/K4]
    g     = 9.81           # gravity [m/s2]
    kappa = 0.4            # von Karman constant
    vi    = 10**-6               # runoff kinetic molecular viscosity [m2/s]
    ch2o  = 4185                 # water specific heat capacity [J/kg/K]
    viw   = 1.48*10**-5          # air kinetic molecular viscosity [m2/s]

class pucmparams:
    prm_asphalt  = {'aG1': 0.20,        'kG1': 0.964446,    'cG1': 1.786005,      'ZmG1': 0.03     , 'G_ManCoef1': 0.05, 'G_hc01': 0.8  /5,   'G_StLayer1': 1.0/5, 'G_slope1': 0.05   }
    prm_concrete = {'aG2': 0.332727,    'kG2': 2.068408,    'cG2': 2.350513 ,     'ZmG2': 0.014508 , 'G_ManCoef2': 0.05, 'G_hc02': 0.205/5,   'G_StLayer2': 1.0/5, 'G_slope2': 0.01   }
    prm_roof1    = {'aR1': 0.42	 ,      'kR1': 3.0 ,        'cR1': 0.7 ,          'ZmR1': 0.01     , 'R_ManCoef1': 0.05, 'R_hc01': 0.11 /5,   'R_StLayer1': 4.1/5, 'R_slope1': 0.01   }
    prm_grass    = {'aG3': 0.456544,    'kG3': 0.4 ,        'cG3':1.0  ,          'ZmG3': 0.01     }

    # Site information ------------------------------------------
    site = 'EquadRoof2010'
    qc   = np.pi/4.0                     # canyon orientation [rad]
    
    # -----------------------------------------------------------
    # 2. specify number and fractions of the different facets ---
    nR = 1                        # roof types: normal||green
    nW = 1                        # wall types: brick||glass
    # can skip concrete, but vegetation parameters need to be present in the last position
    nG = 3                        # ground types: asphalt||concrete||vegetated
    # fractions
    fR = np.array([1])            # fraction for each type of roof
    fW = np.array([1])            # fraction for each type of wall
    fG = np.array([0.5,0.2,0.3])  # fraction for each type of ground
    # for trees
    ft    = 0.0                   # 0.8                  # Same as ftree?
    ftree = 0.0                   # 0.8                  # Fraction of trees in the along-canyon axis
    
    # -----------------------------------------------------------
    # City geometry ---------------------------------------------
    Zr  = 18.9             # roof level (building height)[m]
    Za = 23.23             # reference height [m]
    r  = 0.4               # normalized roof width [m]
    w  = 1-r          # normalized road width [m]
    h   = 0.3              # normalized building height, Ryu (normalize by canyon width)
    H = Zr            # Building height [m]
    W = w / h * H          # Road width [m]
    dR = 0.5               # thickness of roof (m), in SNUUCM, it's = 0.4
    dW = 0.3               # thickness of wall (m), in SNUUCM, it's = 0.4
    # roughness lenght and zero displacement height
    k  = 0.4
    a  = 4.43
    b  = 1.0
    CD = 1.2
    d  = Zr*(1+a**(-r)*(r-1))
    Z0 = Zr*(1-d/Zr)*np.exp(-(0.5*b*CD*(1-d/Zr)*h/k**2)**(-0.5)) # Roughness lenght of city [m]
    
    # -----------------------------------------------------------
    # Roughness lengths for the different facets ----------------
    Zmc = Zr*0.05                                # momentum roughness length above canyon [m], Ryu (10% Zr)
    ZmR = np.array([prm_roof1['ZmR1']])                      # momentum roughness length above roof [m]
    ZmG = np.array([prm_asphalt['ZmG1'], prm_concrete['ZmG2'], prm_grass['ZmG3']])          # momentum roughness length above ground [m], Ryu
    ZmW = np.array([0.01])                      # momentum roughness length on the wall [m]            
    ZhR = ZmR/10                                # heat roughness length above roof [m]
    Zhc = Zmc/10                                # heat roughness length above canyon [m]
    ZhG = ZmG/10                                # heat roughness length above road [m], Ryu
    ZhW = ZmW/10
            
    # -----------------------------------------------------------
    # suface thermal properties: 2nd roof properties are of green roof
    aR = np.array([prm_roof1['aR1']])  # roof surface albedo
    aW = np.array([0.25])              # wall surface albedo
    aG = np.array([prm_asphalt['aG1'], prm_concrete['aG2'], prm_grass['aG3']])  # ground surface albedo
    eR = np.array([0.95])              # roof surface emissivity
    eW = np.array([0.95])              # wall surface emissivity
    eG = np.array([0.95, 0.98, 0.93])  # ground surface emissivity
    aT = 0.2                           # Leaf surface albedo
    eT = 0.95                          # Leaf surface emissivity
    cR = 1e6*np.array([prm_roof1['cR1']])           # heat capacity of roof [J/K/m3]
    cW = 1e6*np.array([1.2])           # heat capacity of wall [J/K/m3]
    cG = 1e6*np.array([prm_asphalt['cG1'], prm_concrete['cG2'], prm_grass['cG3']]) # heat capacity of ground [J/K/m3]
    kR = np.array([prm_roof1['kR1']])               # thermal conductivity of roof [W/K/m]
    kW = np.array([1.3])               # thermal conductivity of wall [W/K/m]
    kG = np.array([prm_asphalt['kG1'], prm_concrete['kG2'], prm_grass['kG3']])     # thermal conductivity of ground [W/K/m]
    alW = kW/cW                        # thermal diffusivity [m2/s]        
    alR = kR/cR                        # thermal diffusivity [m2/s]
    alG = kG/cG                        # thermal diffusivity [m2/s]
        
    # average albedo over all facets of the same surface -------------
    eWe = np.dot(fW,eW)
    eGe = np.dot(fG,eG)
    aWe = np.dot(fW,aW)
    aGe = np.dot(fG,aG)
    
    # soil parameters
    nL  = 10                          # Number of soil layers 
    Ws  = 0.47                        # Volumetric water content at saturation [m3/m3]
    Wr  = 0.15                        # Residual soil water content [m3/m3]
    Ks  = 3.38e-6                     # saturated conductivity [m/s]
    dwG = 0.001                       # depth of water-holding ground pavements [m]
    dwR = 0.001                       # max water holding depth of roof gravel layer [m]
    dvR = 0.1                         # depth of green roof soil [m]
    poR = 0.3                         # porosity of roof gravel [-]
    
    # urban tree and grass parameters
    aleaf            = 1.5                   # tree crown ratio [m]
    htree            = 7.3                   # Tree height [m] - soil to mid crown
    dtree            = 2.2                   # tree-wall distance [m]
    LAItree          = 4                     # tree leaf area index
    Aleaf            = LAItree * 2.0 * aleaf # Crown surface area
    LAIgrass         = 1.5                   # leaf area index for short grass
    max_water        = 1.                    # maximum water holding amount [kg/m2], Young-Hee Ryu
    zroot            = 1.0                   # Root depth [m]
    # Croot = 3.67                           # f value in Jarvis (1989), JH
    Croot            = 2.5                   # root distribution parameter; f value in Jarvis (1989), JH
    qc1              = 0.6                   # Critical value of the normalized soil water content
    
    # -----------------------------------------------------------
    # Initializing surface temperatures and soil moisture
    TG1  = 17.0  
    TG3  = 17.0  
    TG4  = 17.0  
    Tsi  = 25.0 # initial roof/wall temperature from EC           
    TGi  = np.array([TG1, (TG3+TG4)/2.0, TG4])
    qmi  = 0.2 #qmi_start   # water available at the roof (green roof)
    qmib = qmi*1.5
        
    # Initial moisture profile
    zlay             = np.arange(1,6,1)  
    smc_profile      = np.zeros(nL)
    smc_profile[:5]  = ( (qmib - qmi) /4 )*(zlay-5) + qmib
    smc_profile[5:]  = qmib
    # Wang et al starts the profile as a uniform soil moisture in z
    #smc_profile[:]   = qmib  # 0.06 sunny-cloudy, 0.2 rainy1, 0.145 rainy2
    
    # -----------------------------------------------------------
    # Discretization of ground soil for hydrological modeling
    dg     = 2.0 / nL   
    dG     = dg * np.ones(nL)                                # thickness below the soil surface of layer i [m]
    dgG    = 2.0 / nL * np.ones(nL)                          # vegetated ground  
    zdepth = np.linspace( dg/2, 2.0 - dg/2 , 10)             # Mid point depth below the soil surface [m]
    rootl  = Croot*(dg/zroot)*np.exp(-Croot*(zdepth/zroot))  # Root length density function for layer i 
    
    # -----------------------------------------------------------
    # Sky view factors Fji from j to i --------------------------
    # G - ground
    # W - wall
    # S - sky
    # R - roof
    FGS = 0.4186
    FGT = 0.1539
    FGW = 1. - FGS - FGT
    
    FWS = 0.2392
    FWG = FWS
    FWT = 0.2700
    FWW = 1. - FWS - FWG - FWT
        
    FTS = 0.2897
    FTW = 0.3529
    FTT = 0.0489
    FTG = 1. - FTW - FTS - FTT
    
    # -----------------------------------------------------------
    # Runoff variables ------------------------------------------
    #beta             = 4                    # nondimensional factor that includes turbulent effects
    #DWater_Molecular = beta * 0.143e-6      # molecular diffusivity in water
    #minRunoffVel     = 1/1000               # Minimun runoff velocity [m/s]
    
    # Roof -------------
    #statLayer_rof    = dwR
    #hc0_rof          = 0.5/1000 #0.5/1000           # minimum layer of water before runoff model is activated [m]
    #s0_rof           = 0.001                        # asphalt slope, %
    #Nroof            = 200                           # number of layers for ground heat flux equation (discrete solution)
    #Lz_roof          = dR                           # soil depth where heat equation is solved, [m]
    #dz_roof          = Lz_roof/Nroof                # dz of soil, [m]
    
    # Ground -------------
    # Asphalt
    #statLayer_asp    = dwG
    #hc0_asp          = 0.5/1000 #0.5/1000           # minimum layer of water before runoff model is activated [m]
    #s0_asp           = 0.001                        # asphalt slope, %
    #Nground_asp      = 200                           # number of layers for ground heat flux equation (discrete solution)
    #Lz_ground_asp    = 0.15                         # soil depth where heat equation is solved, [m]
    #dz_ground_asp    = Lz_ground_asp/Nground_asp    # dz of soil, [m]
    
    # Concrete
    #statLayer_con    = dwG
    #hc0_con          = 0.5/1000 #0.5/1000         # minimum layer of water before runoff model is activated [m]
    #s0_con           = 0.001                      # asphalt slope, %
    #Nground_con      = 200                         # number of layers for ground heat flux equation (discrete solution)
    #Lz_ground_con    = 0.2                        # soil depth where heat equation is solved, [m]
    #dz_ground_con    = Lz_ground_con/Nground_con  # dz of soil, [m]
            
    # List of parameters for runoff model (ground)
    G_hc0         = [ prm_asphalt['G_hc01']/1000,     prm_concrete['G_hc02']/1000     ]
    G_StLayer     = [ prm_asphalt['G_StLayer1']/1000, prm_concrete['G_StLayer2']/1000 ]    
    G_slope       = [ prm_asphalt['G_slope1'],   prm_concrete['G_slope2']   ]
    G_surf_length = [      1.,     10.     ]
    G_ManCoef     = [ prm_asphalt['G_ManCoef1'], prm_concrete['G_ManCoef2'] ]
        
    # List of parameters for runoff roof
    R_hc0         = [ prm_roof1['R_hc01']/1000      ]
    R_StLayer     = [ prm_roof1['R_StLayer1']/1000  ]
    R_slope       = [ prm_roof1['R_slope1']    ]
    R_surf_length = [   10.                    ]
    R_ManCoef     = [ prm_roof1['R_ManCoef1']  ]