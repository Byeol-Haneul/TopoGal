import os
import sys
import h5py 
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.machine import *

'''
Note:
Before running preprocessing scripts, please be aware that the master settings (MACHINE, TYPE, SUBGRID) are appropriately set.
'''

# ---- CONSTANTS ---- #
DIM = 3

if TYPE == "Quijote" or TYPE == "Quijote_Rockstar" or TYPE == "fR":
    BOXSIZE = 1e3 
elif TYPE == "CAMELS" or TYPE == "CAMELS_SB28":
    BOXSIZE = 25e3
elif TYPE == "CAMELS_50":
    BOXSIZE = 50e3
else:
    raise Exception("Invalid Simulation Suite")

if TYPE == "CAMELS" or TYPE == "CAMELS_SB28":
    MASS_UNIT = 1e10
    Nstar_th = 20 
    MASS_CUT = 2e8
elif TYPE == "CAMELS_50":
    MASS_UNIT = 1e10
    Nstar_th = 40
    MASS_CUT = 4e8
else:
    raise Exception("Invalid Simulation Suite")

modes = {"ISDISTANCE": 1, "ISAREA": 2, "ISVOLUME": 3}
global_centroid = None # to be updated.

# --- HYPERPARAMS --- #
#dense: 0.02, sparse: 0.01, fiducial: 0.015
#r_link = 0.015 
r_link = float(os.getenv("R_LINK", 0.015))
NUMCUT = int(os.getenv("NUMCUT", 10000))
print(f"THE R_LINK IS SET AS {r_link}", file=sys.stderr)
print(f"THE NUMCUT IS SET AS {NUMCUT}", file=sys.stderr)

MINCLUSTER = 5 
NUMPOINTS  = -1
NUMEDGES   = NUMCUT if (TYPE == "CAMELS_50" or TYPE == "CAMELS_SB28") else -1
NUMTETRA   = NUMCUT if (TYPE == "Quijote" or TYPE == "Quijote_Rockstar" or TYPE == "fR" or TYPE == "CAMELS_50" or TYPE == "CAMELS_SB28") else -1

## OPTIONS ##
ENABLE_PROFILING = False
#############

if TYPE == "Quijote" or TYPE == "Quijote_Rockstar" or TYPE == "fR":# or TYPE == "CAMELS_50":
    in_dir = BASE_DIR + "sims/"
    cc_dir = DATA_DIR + f"cc_{NUMTETRA}/"
    tensor_dir = DATA_DIR + f"tensors_{NUMTETRA}/"
elif TYPE == "CAMELS_SB28":
    in_dir = BASE_DIR + f"sims/"
    cc_dirs_option = {0.03: "cc/", 0.05: "cc_dense/", 0.04: "cc_sparse/"}
    tensor_dirs_option = {0.03: "tensors/", 0.05: "tensors_dense/", 0.04: "tensors_sparse/"}
    cc_dir = DATA_DIR + cc_dirs_option.get(r_link, "")
    tensor_dir = DATA_DIR + tensor_dirs_option.get(r_link, "")
else:
    in_dir = BASE_DIR + f"sims/"
    cc_dirs_option = {0.015: "cc/", 0.02: "cc_dense/", 0.01: "cc_sparse/"}
    tensor_dirs_option = {0.015: "tensors/", 0.02: "tensors_dense/", 0.01: "tensors_sparse/"}
    cc_dir = DATA_DIR + cc_dirs_option.get(r_link, "")
    tensor_dir = DATA_DIR + tensor_dirs_option.get(r_link, "")

os.makedirs(cc_dir, exist_ok=True)
os.makedirs(tensor_dir, exist_ok=True)

def load_catalog(directory, filename):
    '''
    Modified from CosmoGraphNet
    arXiv:2204.13713
    https://github.com/PabloVD/CosmoGraphNet/
    '''
    if TYPE == "Quijote" or TYPE == "Quijote_Rockstar" or TYPE == "fR":
        pos = np.loadtxt(directory + filename)/BOXSIZE
    else:
        f = h5py.File(directory + filename, 'r')
        pos   = f['/Subhalo/SubhaloPos'][:]/BOXSIZE
        Mstar = f['/Subhalo/SubhaloMassType'][:,4] * MASS_UNIT
        Rstar = f["Subhalo/SubhaloHalfmassRadType"][:,4]
        Nstar = f['/Subhalo/SubhaloLenType'][:,4] 
        Metal = f["Subhalo/SubhaloStarMetallicity"][:]
        Vmax = f["Subhalo/SubhaloVmax"][:]
        f.close()
    
    # Some simulations are slightly outside the box, correct it
    pos[np.where(pos<0.0)]+=1.0
    pos[np.where(pos>1.0)]-=1.0

    if TYPE == "Quijote" or TYPE == "Quijote_Rockstar" or TYPE == "fR":
        feat = np.zeros(pos.shape)
    else:
        indexes = np.where(Nstar>Nstar_th)[0]
        #indexes = np.where(Mstar>MASS_CUT)[0]
        pos     = pos[indexes]
        Mstar   = Mstar[indexes]
        Rstar   = Rstar[indexes]
        Metal   = Metal[indexes]
        Vmax    = Vmax[indexes]

        #Normalization
        Mstar = np.log10(1.+ Mstar)
        Rstar = np.log10(1.+ Rstar)
        Metal = np.log10(1.+ Metal)
        Vmax  = np.log10(1.+ Vmax)

        feat = np.vstack((Mstar, Rstar, Metal, Vmax)).T

    return pos, feat

def normalize(value, option):
    power = modes[option]
    return value / (r_link ** power)