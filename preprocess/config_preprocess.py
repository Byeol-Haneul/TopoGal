import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.machine import BASE_DIR

# ---- CONSTANTS ---- #
BOXSIZE = 25e3
DIM = 3
MASS_UNIT = 1e10
Nstar_th = 20 # not used
MASS_CUT = 2e8
modes = {"ISDISTANCE": 1, "ISAREA": 2, "ISVOLUME": 3}

global_centroid = None # to be updated.

# --- HYPERPARAMS --- #
r_link = 0.015
MINCLUSTER = 5 #>10 Found no clusters made in some catalogs. 
NUMPOINTS = -1
NUMEDGES = -1
NUMTETRA = -1

## OPTIONS
ENABLE_PROFILING = False

in_dir = BASE_DIR + "/IllustrisTNG/data/"
cc_dir = BASE_DIR + "/IllustrisTNG/combinatorial/cc/"
tensor_dir = BASE_DIR + "/IllustrisTNG/combinatorial/tensors/"
pad_dir = BASE_DIR + "/IllustrisTNG/combinatorial/padded/"
label_filename = BASE_DIR + "/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"

# Create the directories if they don't exist
os.makedirs(cc_dir, exist_ok=True)
os.makedirs(tensor_dir, exist_ok=True)
os.makedirs(pad_dir, exist_ok=True)

## HELPER FUNCTION! ##
def normalize(value, option):
    power = modes[option]
    return value / (r_link)


feature_sets = [
    'x_0', 'x_1', 'x_2', 'x_3', 'x_4',

    'n0_to_0', 'n1_to_1', 'n2_to_2', 'n3_to_3', 'n4_to_4',
    'n0_to_1', 'n0_to_2', 'n0_to_3', 'n0_to_4',
    'n1_to_2', 'n1_to_3', 'n1_to_4',
    'n2_to_3', 'n2_to_4',
    'n3_to_4',

    'euclidean_0_to_0', 'euclidean_1_to_1', 'euclidean_2_to_2', 'euclidean_3_to_3', 'euclidean_4_to_4',
    'euclidean_0_to_1', 'euclidean_0_to_2', 'euclidean_0_to_3', 'euclidean_0_to_4',
    'euclidean_1_to_2', 'euclidean_1_to_3', 'euclidean_1_to_4',
    'euclidean_2_to_3', 'euclidean_2_to_4',
    'euclidean_3_to_4',

    'global_feature'
]