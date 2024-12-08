import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.machine import *

# ---- CONSTANTS ---- #
if TYPE == "BISPECTRUM" or TYPE == "fR":
    BOXSIZE = 1e3 
elif TYPE == "CAMELS":
    BOXSIZE = 25e3
else:
    raise Exception("Invalid Simulation Suite")

DIM = 3
MASS_UNIT = 1e10
Nstar_th = 20 
MASS_CUT = 2e8
modes = {"ISDISTANCE": 1, "ISAREA": 2, "ISVOLUME": 3}
global_centroid = None # to be updated.

# --- HYPERPARAMS --- #
r_link = 0.02 #dense: 0.02, sparse: 0.01, fiducial: 0.015
MINCLUSTER = 5 #>10 Found no clusters made in some catalogs.
NUMPOINTS  = -1
NUMEDGES   = -1
NUMTETRA   = 3000 if (TYPE == "BISPECTRUM" or TYPE == "fR") else -1

## OPTIONS
ENABLE_PROFILING = False

if TYPE == "BISPECTRUM" or TYPE == "fR":
    in_dir = BASE_DIR + "sims/"
    cc_dir = DATA_DIR + f"cc_{NUMTETRA}/"
    tensor_dir = DATA_DIR + f"tensors_{NUMTETRA}/"
else:
    in_dir = BASE_DIR + f"sims/"
    cc_dirs_option = {0.015: "cc/", 0.02: "cc_dense/", 0.01: "cc_sparse/"}
    tensor_dirs_option = {0.015: "tensors/", 0.02: "tensors_dense/", 0.01: "tensors_sparse/"}
    cc_dir = DATA_DIR + cc_dirs_option.get(r_link, "")
    tensor_dir = DATA_DIR + tensor_dirs_option.get(r_link, "")

# Create the directories if they don't exist
os.makedirs(cc_dir, exist_ok=True)
os.makedirs(tensor_dir, exist_ok=True)

## HELPER FUNCTION! ##
def normalize(value, option):
    power = modes[option]
    return value / (r_link ** power)