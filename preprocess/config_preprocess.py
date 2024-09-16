import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.machine import BASE_DIR

# ---- CONSTANTS ---- #
BOXSIZE = 25e3
DIM = 3
MASS_UNIT = 1e10
Nstar_th = 20 # not used
MASS_CUT = 1e8
modes = {"ISDISTANCE": 1, "ISAREA": 2, "ISVOLUME": 3}

global_centroid = None # to be updated.

# --- HYPERPARAMS --- #
r_link = 0.015
MINCLUSTER = 5 #>10 Found no clusters made in some catalogs. 
NUMPOINTS = -1
NUMEDGES = -1
NUMTETRA = 500

## OPTIONS
ENABLE_PROFILING = False

in_dir = BASE_DIR + "/IllustrisTNG/data/"
cc_dir = BASE_DIR + "/IllustrisTNG/combinatorial/cc/"
tensor_dir = BASE_DIR + "/IllustrisTNG/combinatorial/tensors/"
label_filename = BASE_DIR + "/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"

# Create the directories if they don't exist
os.makedirs(cc_dir, exist_ok=True)
os.makedirs(tensor_dir, exist_ok=True)

## HELPER FUNCTION! ##
def normalize(value, option):
    power = modes[option]
    return value / (r_link)