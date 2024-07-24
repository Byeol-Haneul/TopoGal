from mpi4py import MPI
import numpy as np
import h5py
import random
from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import Delaunay, distance
from toponetx.classes.simplicial_complex import SimplicialComplex
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

from toponetx.readwrite.serialization import to_pickle
from toponetx.readwrite.serialization import load_from_pickle

# --- CONSTANTS ---#
BOXSIZE = 25e3
# -----------------#

def load_catalog(directory, filename):
    f = h5py.File(directory+filename,'r')
    pos   = f['/Subhalo/SubhaloPos'][:]/BOXSIZE
    Mstar = f['/Subhalo/SubhaloMassType'][:,4] #Msun/h
    Rstar = f["Subhalo/SubhaloHalfmassRadType"][:,4]
    feat = np.vstack((Mstar, Rstar)).T
    feat = np.hstack((pos, feat))
    return pos, feat

def create_simplical_complex(pos, feat):
    tri = Delaunay(pos)  # Create Delaunay triangulation
    simplices = []
    indices_included = set()
    for simplex in tri.simplices:
        simplices.append(simplex)
        indices_included |= set(simplex)

    # Re-index vertices before constructing the simplicial complex
    idx_dict = {i: j for j, i in enumerate(indices_included)}
    for i in range(len(simplices)):
        for j in range(3):
            simplices[i][j] = idx_dict[simplices[i][j]]

    sc = SimplicialComplex(simplices)
    coords = pos[list(indices_included)]

    # ADD NODE & EDGE FEATURES
    node_data = {num: feat[num] for num in range(feat.shape[0])}
    sc.set_simplex_attributes(node_data, name="node_feat")
    edge_data = {(start, end): [np.linalg.norm(pos[start] - pos[end])] for start, end in sc.skeleton(1)}
    edge_data.update({(end, start): [np.linalg.norm(pos[start] - pos[end])] for start, end in sc.skeleton(1)})
    sc.set_simplex_attributes(edge_data, name="edge_feat")
    print(sc)
    return sc, coords


def main():
    # MPI Initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    in_dir = "/data2/jylee/topology/IllustrisTNG/data/"
    out_dir = "/data2/jylee/topology/IllustrisTNG/simplex/"

    # Distribute the tasks
    total_jobs = 1000
    jobs_per_process = total_jobs // size
    extra_jobs = total_jobs % size

    if rank < extra_jobs:
        start_num = rank * (jobs_per_process + 1)
        end_num = start_num + jobs_per_process + 1
    else:
        start_num = rank * jobs_per_process + extra_jobs
        end_num = start_num + jobs_per_process

    for num in range(start_num, end_num):
        in_filename = f"data_{num}.hdf5"
        out_filename = f"data_{num}.pickle"

        pos, feat = load_catalog(in_dir, in_filename)
        sc, coords = create_simplical_complex(pos, feat)

        print(f"Process {rank}: Created simplicial complex for file {in_filename}")
        to_pickle(sc, out_dir + out_filename)

    # Finalize MPI
    comm.Barrier()
    MPI.Finalize()

if __name__ == "__main__":
    main()