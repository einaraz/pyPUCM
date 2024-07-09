from datetime import datetime

periods_to_optimize = {      'sunny':  [datetime(2010,8,28),  datetime(2010,9,1)],
                        'sunny_gray':  [datetime(2010,8,28),  datetime(2010,9,5)],
                            'rainy1':  [datetime(2010,7,12),  datetime(2010,7,15)],
                            'rainy2':  [datetime(2010,7,24),  datetime(2010,7,27)]  }

periods_qmi = {'sunny': 0.25, 
               'sunny_gray': 0.25, 
               'rainy1': 0.27, 
               'rainy2': 0.23}

facet_shift = {'TG1': 20, 
               'TG2': 60, 
               'TG3': 20, 
               'TR1': 20}

facet_name = { 'TG1': 'G, asphalt', 
               'TG2': 'G, concrete', 
               'TG3': 'G, grass', 
               'TR1':  'Roof' }

# sunny: ~ 1 minute to simulate at 20 s
optionsopt = { 
           # sunny: for all facets, change flux model
           'case01': {   'case_opt': 'sunny',      'opt_var': 'TG1', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },
           'case02': {   'case_opt': 'sunny',      'opt_var': 'TG2', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },
           'case03': {   'case_opt': 'sunny',      'opt_var': 'TG3', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },
           'case04': {   'case_opt': 'sunny',      'opt_var': 'TR1', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },
           'case05': {   'case_opt': 'sunny',      'opt_var': 'TG1', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
           'case06': {   'case_opt': 'sunny',      'opt_var': 'TG2', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
           'case07': {   'case_opt': 'sunny',      'opt_var': 'TG3', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
           'case08': {   'case_opt': 'sunny',      'opt_var': 'TR1', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
           # sunny_gray: for all facets, change flux model
           'case09': {   'case_opt': 'sunny_gray', 'opt_var': 'TG1', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },
           'case10': {   'case_opt': 'sunny_gray', 'opt_var': 'TG2', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },
           'case11': {   'case_opt': 'sunny_gray', 'opt_var': 'TG3', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },
           'case12': {   'case_opt': 'sunny_gray', 'opt_var': 'TR1', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },
           'case13': {   'case_opt': 'sunny_gray', 'opt_var': 'TG1', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
           'case14': {   'case_opt': 'sunny_gray', 'opt_var': 'TG2', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
           'case15': {   'case_opt': 'sunny_gray', 'opt_var': 'TG3', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
           'case16': {   'case_opt': 'sunny_gray', 'opt_var': 'TR1', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },

           # sunny: for all facets, change flux model
           'case17': {   'case_opt': 'sunny',      'opt_var': 'TG1', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },
           'case18': {   'case_opt': 'sunny',      'opt_var': 'TG2', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },
           'case19': {   'case_opt': 'sunny',      'opt_var': 'TG3', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },
           'case20': {   'case_opt': 'sunny',      'opt_var': 'TR1', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },
           'case21': {   'case_opt': 'sunny',      'opt_var': 'TG1', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
           'case22': {   'case_opt': 'sunny',      'opt_var': 'TG2', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
           'case23': {   'case_opt': 'sunny',      'opt_var': 'TG3', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
           'case24': {   'case_opt': 'sunny',      'opt_var': 'TR1', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
           # sunny_gray: for all facets, change flux model
           'case25': {   'case_opt': 'sunny_gray', 'opt_var': 'TG1', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },
           'case26': {   'case_opt': 'sunny_gray', 'opt_var': 'TG2', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },  
           'case27': {   'case_opt': 'sunny_gray', 'opt_var': 'TG3', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },  
           'case28': {   'case_opt': 'sunny_gray', 'opt_var': 'TR1', 'runoffmodel': 'HERB',  'TurbFlux': 'PUCM' },  
           'case29': {   'case_opt': 'sunny_gray', 'opt_var': 'TG1', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
           'case30': {   'case_opt': 'sunny_gray', 'opt_var': 'TG2', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },  
           'case31': {   'case_opt': 'sunny_gray', 'opt_var': 'TG3', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
           'case32': {   'case_opt': 'sunny_gray', 'opt_var': 'TR1', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
           
           # rainy1: for all facets, change flux model
            'case33': {   'case_opt': 'rainy1',     'opt_var': 'TG1', 'runoffmodel': 'HAMID',  'TurbFlux': 'MOST' }, #
            'case34': {   'case_opt': 'rainy1',     'opt_var': 'TG2', 'runoffmodel': 'HAMID',  'TurbFlux': 'MOST' },
            'case35': {   'case_opt': 'rainy1',     'opt_var': 'TR1', 'runoffmodel': 'HAMID',  'TurbFlux': 'MOST' }, #
            'case36': {   'case_opt': 'rainy1',     'opt_var': 'TG1', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },  #
            'case37': {   'case_opt': 'rainy1',     'opt_var': 'TG2', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
            'case38': {   'case_opt': 'rainy1',     'opt_var': 'TR1', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },  #
            # rainy2: for all facets, change flux model
            'case39': {   'case_opt': 'rainy2',     'opt_var': 'TG1', 'runoffmodel': 'HAMID',  'TurbFlux': 'MOST' },
            'case40': {   'case_opt': 'rainy2',     'opt_var': 'TG2', 'runoffmodel': 'HAMID',  'TurbFlux': 'MOST' }, #
            'case41': {   'case_opt': 'rainy2',     'opt_var': 'TR1', 'runoffmodel': 'HAMID',  'TurbFlux': 'MOST' },
            'case42': {   'case_opt': 'rainy2',     'opt_var': 'TG1', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
            'case43': {   'case_opt': 'rainy2',     'opt_var': 'TG2', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' },
            'case44': {   'case_opt': 'rainy2',     'opt_var': 'TR1', 'runoffmodel': 'HERB',  'TurbFlux': 'MOST' }
           }

testoptions = {
           '49July2010': {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,7,4),  datetime(2010,7,9)],  'qmi_start': 0.42 },
           '49May2010':  {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,5,4),  datetime(2010,5,9)],  'qmi_start': 0.42 },
           'July2010':   {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,7,1),  datetime(2010,7,30)], 'qmi_start': 0.16 },  # original
           'July2010newlg':   {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,7,1),  datetime(2010,7,30)], 'qmi_start': 0.16 }, # same as original, but corrected longwave
           'July2010fixh1':   {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,7,1),  datetime(2010,7,30)], 'qmi_start': 0.16 }, # evaporates when runoff_height>0
           'July2010fixh2':   {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,7,1),  datetime(2010,7,30)], 'qmi_start': 0.16 }, # Only drops h to zero if no rain in previous 24 hours
           'July2010fixh3':   {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,7,1),  datetime(2010,7,30)], 'qmi_start': 0.16 }, # Only drops h to zero if no rain in previous 6 hours
           'July2010newlgx':   {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,7,1),  datetime(2010,7,30)], 'qmi_start': 0.16 }, # same as original, but corrected longwave
           'model1':   {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,7,9),  datetime(2010,7,17)], 'qmi_start': 0.25 },  # original
           'model2':   {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,7,9),  datetime(2010,7,17)], 'qmi_start': 0.25 },  # original
           'model3':   {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,7,9),  datetime(2010,7,17)], 'qmi_start': 0.25 },  # original
           'artificial0':   {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,6,25), datetime(2010,7,7)], 'qmi_start': 0.25 },  # no rain in this period
           'artificial1':   {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,6,25), datetime(2010,7,7)], 'qmi_start': 0.25 },  # add artificial rain
           'artificial2':   {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,6,25), datetime(2010,7,7)], 'qmi_start': 0.25 },  # add artificial rain - only runoff, no thermal exchange
           'artificial3':   {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,6,25), datetime(2010,7,7)], 'qmi_start': 0.25 },  # add artificial rain + very low hf (when runoff starts)
           'equad2010': {  'runoffmodel': 'HAMID', 'TurbFlux': 'MOST', 'period': [datetime(2010,5,1),  datetime(2010,10,1)],    'qmi_start': 0.38 },
              }