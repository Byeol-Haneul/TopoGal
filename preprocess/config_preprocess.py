import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.machine import *

# ---- CONSTANTS ---- #
BOXSIZE = 1e3 if TYPE == "BISPECTRUM" else 25e3
DIM = 3
MASS_UNIT = 1e10
Nstar_th = 20 
MASS_CUT = 2e8
modes = {"ISDISTANCE": 1, "ISAREA": 2, "ISVOLUME": 3}
global_centroid = None # to be updated.

# --- HYPERPARAMS --- #
r_link = 0.01 #dense: 0.02, sparse: 0.01, fiducial: 0.015
MINCLUSTER = 5 #>10 Found no clusters made in some catalogs.
NUMPOINTS  = -1
NUMEDGES   = 5000 if TYPE == "BISPECTRUM" else -1
NUMTETRA   = 5000 if TYPE == "BISPECTRUM" else -1

## OPTIONS
ENABLE_PROFILING = False

if TYPE == "BISPECTRUM":
    in_dir = BASE_DIR + "new/"
    cc_dir = DATA_DIR + "cc_5000/"
    tensor_dir = DATA_DIR + "tensors_5000/"
else:
    in_dir = BASE_DIR + f"/data/"
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
    return value / (r_link)