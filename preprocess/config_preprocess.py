import os

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
MINCLUSTER = 7 #>10 Found no clusters made in some catalogs. 

## OPTIONS
ENABLE_PROFILING = True

# Define your directories and file paths
in_dir = "/data2/jylee/topology/IllustrisTNG/data/"
cc_dir = "/data2/jylee/topology/IllustrisTNG/combinatorial/cc_test/"
tensor_dir = "/data2/jylee/topology/IllustrisTNG/combinatorial/tensors_test/"
label_filename = "/data2/jylee/topology/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"

# Create the directories if they don't exist
os.makedirs(cc_dir, exist_ok=True)
os.makedirs(tensor_dir, exist_ok=True)

## HELPER FUNCTION! ##
def normalize(value, option):
    power = modes[option]
    return value / (r_link)