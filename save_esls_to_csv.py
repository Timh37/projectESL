'''
@author: Tim Hermans
t(dot)h(dot)j(dot)hermans@uu(dot)nl
'''
import os
import sys
sys.path.insert(1, 'projectESL/')

from preprocessing import ingest_GESLA3_files,detrend_gesla_dfs, deseasonalize_gesla_dfs, subtract_amean_from_gesla_dfs
from esl_analysis import pot_extremes_from_gesla_dfs
from I_O import load_config

output_dir = '/Users/timhermans/Documents/PostDoc/Phase2_esl_sensitivity/ESLs_for_SJewson/detrended_deseasonalized_dmax'
cfg = load_config('config.yml')
preproc_settings = cfg['preprocessing']
fns = ['grand_isle-8761724-usa-noaa','hoek_van_holland-hoekvhld-nld-rws','sheerness-she-gbr-bodc','the_battery-8518750-usa-noaa']
dfs = ingest_GESLA3_files(cfg['input']['paths']['gesla3'],preproc_settings,fns=fns)
                    
#preproc options:
if preproc_settings['detrend']:
    dfs = detrend_gesla_dfs(dfs)
if preproc_settings['deseasonalize']:
    dfs = deseasonalize_gesla_dfs(dfs)
if preproc_settings['subtract_amean']:
    dfs = subtract_amean_from_gesla_dfs(dfs)
                
extremes = pot_extremes_from_gesla_dfs(dfs,preproc_settings['extremes_threshold'],preproc_settings['declus_method'],preproc_settings['declus_window'])
for k,v in extremes.items():
    v.to_csv(os.path.join(output_dir,k+'_extremes.csv'))