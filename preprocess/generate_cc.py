from mpi4py import MPI
import numpy as np
import h5py
import random
from itertools import product, combinations
import sys

from scipy.spatial import Delaunay, distance
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import networkx as nx
from toponetx.classes.simplicial_complex import SimplicialComplex
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.readwrite.serialization import to_pickle
from toponetx.readwrite.serialization import load_from_pickle


# --- CONSTANTS --- #
BOXSIZE = 25e3
DIM = 3

# NUMBER OF x-CELLS #
# NUMPOINTS [0-cell] : Number of points. Use -1 for all points.
# NUMEDGES  [1-cell] : Number of edges. Use -1 for all edges.
# NUMTETRA  [2-cell] : Number of tetrahedra. Use -1 for all tetrahedra.
# NUMCLUSTER currently not defined #
NUMPOINTS = -1
NUMEDGES = -1
NUMTETRA = -1
# ------------------ #

def load_catalog(directory, filename):
    f = h5py.File(directory+filename,'r')
    pos   = f['/Subhalo/SubhaloPos'][:]/BOXSIZE
    Mstar = f['/Subhalo/SubhaloMassType'][:,4] #Msun/h
    Rstar = f["Subhalo/SubhaloHalfmassRadType"][:,4]
    feat = np.vstack((Mstar, Rstar)).T
    feat = np.hstack((pos, feat))

    ## REDEFINE GLOBAL VARIABLES ##
    global NUMPOINTS
    NUMPOINTS = pos.shape[0]
    pos = pos[:,:DIM]

    return pos, feat

class Tetrahedron:
    def __init__(self, index, nodes, pos):
        self.index = index
        self.nodes = nodes
        self.pos = np.array([pos[node] for node in nodes])
        self.volume = self.calculate_volume()
        self.midpoint = self.calculate_midpoint()

    def calculate_volume(self):
        mat = np.zeros([4, 4])
        mat[:, :3] = self.pos
        mat[:, 3] = 1.0
        return np.abs(np.linalg.det(mat)) / 6

    def calculate_midpoint(self):
        return np.mean(self.pos, axis=0)

def get_tetrahedra(pos, feat):
    # Generate Delaunay triangulation
    tri = Delaunay(pos)

    # Calculate volumes and store tetrahedra
    tetrahedra = []
    for i, simplex in enumerate(tri.simplices):
        tetra = Tetrahedron(i, simplex, pos)
        tetrahedra.append(tetra)

    # Sort tetrahedra by volume (in increasing order)
    tetrahedra.sort(key=lambda x: x.volume)

    # Scale the volumes to make an embedding
    volumes = np.array([tetra.volume for tetra in tetrahedra])
    log_volumes = np.log10(volumes + 1e-20)  # Add small value to avoid log(0)
    volume_scaler = StandardScaler()
    scaled_volumes = volume_scaler.fit_transform(log_volumes.reshape(-1, 1)).flatten()
    print("""[LOG] GENERATED TETRA""", file=sys.stderr)
    return tetrahedra, scaled_volumes

def get_tetra_edges(tetra):
    edges = set()
    distances = []
    for (i, j) in combinations(range(4), 2):
        edge = tuple(sorted((tetra.nodes[i], tetra.nodes[j])))
        distance = np.linalg.norm(tetra.pos[i] - tetra.pos[j])
        edges.add(edge)
        distances.append(distance)
    return edges, distances

def get_edges(tetrahedra):
    all_edges = set()
    edge_distances = {}
    for tetra in tetrahedra:
        edges, distances = get_tetra_edges(tetra)
        for edge, distance in zip(edges, distances):
            all_edges.add(edge)
            edge_distances[edge] = distance
    return all_edges, edge_distances


def create_cc(pos, feat):
    global NUMEDGES, NUMTETRA 
    
    # 1. Get Tetrahedra
    tetrahedra, scaled_volumes = get_tetrahedra(pos, feat)

    # 2. Get edges
    all_edges, edge_distances = get_edges(tetrahedra)

    # 3. Clustering on Tetrahedra
    clusters = clustering(tetrahedra, scaled_volumes)

    if NUMEDGES == -1:
        NUMEDGES = len(all_edges)
    if NUMTETRA == -1:
        NUMTETRA = len(tetrahedra)

    # 4. Generate Combinatorial Complex
    print(f"""[LOG] We will select {NUMEDGES} edges and {NUMTETRA} tetra""", file=sys.stderr)
    cc = CombinatorialComplex()

    ## 4-1 ADD NODES ##
    for node, node_data in zip(list(range(NUMPOINTS)), feat):
        cc.add_cell([node], rank=0)

    node_data = {num: feat[num] for num in range(NUMPOINTS)}

    ## 4-2 ADD EDGES ##
    edges_with_distances = [(edge, distance) for edge, distance in edge_distances.items()]
    edges_with_distances.sort(key=lambda x: x[1])
    sorted_edges = [edge for edge, distance in edges_with_distances][:NUMEDGES]
    sorted_distances = [distance for edge, distance in edges_with_distances][:NUMEDGES]

    for edge, distance in zip(sorted_edges, sorted_distances):
        assert len(edge) == 2
        cc.add_cell(edge, rank=1)
    
    edge_data = {edge: data for edge, data in zip(sorted_edges, sorted_distances)}


    ## 4-3 ADD TETRA ##
    tetrahedra.sort(key=lambda x: x.volume)
    sorted_tetra = tetrahedra[:NUMTETRA]

    for tetra in sorted_tetra:
        assert len(list(tetra.nodes)) == 4
        cc.add_cell(list(tetra.nodes), rank=2)

    tetra_data = {tuple(tetra.nodes): tetra.volume for tetra in sorted_tetra}

    ## 4-4 ADD Clusters ##
    for cluster in clusters.values():
        assert len(list(cluster['node_set'])) > 4
        cc.add_cell(list(cluster['node_set']), rank=3)


    cluster_data = {tuple(cluster['node_set']): cluster['merged_vol'] for cluster in clusters.values()}


    # ADD CELL ATTRIBUTES
    cc.set_cell_attributes(node_data, name="node_feat")
    cc.set_cell_attributes(edge_data, name="edge_feat")
    cc.set_cell_attributes(tetra_data, name="tetra_feat")
    cc.set_cell_attributes(cluster_data, name="cluster_feat")
    return cc


def clustering(tetrahedra, scaled_volumes):
    # Perform DBSCAN clustering on the midpoints
    embeddings = np.array([np.append(tetra.midpoint, scaled_volumes[i]) for i, tetra in enumerate(tetrahedra)])

    # Add a Dimension of volume (to combine similar scales)
    db = DBSCAN(eps=0.05, min_samples=2).fit(embeddings)
    labels = db.labels_

    # Merge tetrahedra within each cluster
    clusters = {}
    for label in set(labels):
        if label == -1:
            # Ignore noise points
            continue
        indices = np.where(labels == label)[0]
        merged_tetra = []
        merged_vol = 0
        node_set = set()
        
        for idx in indices:
            tetra = tetrahedra[idx]
            merged_tetra.append(tetra)
            merged_vol += tetra.volume
            node_set.update(tetra.nodes)

        clusters[label] = {
            'merged_tetra': merged_tetra,
            'merged_vol': merged_vol,
            'node_set': node_set
        }

    print(f"""
    [LOG] We Currently have {len(embeddings)} Tetrahedra.
    [LOG] Generated {len(clusters)} Clusters of Tetrahedra. 
    [LOG] Mean number of nodes per cluster is {np.mean([len(clusters[idx]['node_set']) for idx in range(len(clusters))])}
    [LOG] Max number of nodes per cluster is {np.max([len(clusters[idx]['node_set']) for idx in range(len(clusters))])} and the number is {np.argmax([len(clusters[idx]['node_set']) for idx in range(len(clusters))])}""", file=sys.stderr)
    
    return clusters

def main():
    # MPI Initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    in_dir = "/data2/jylee/topology/IllustrisTNG/data/"
    out_dir = "/data2/jylee/topology/IllustrisTNG/combinatorial/cc_extended/"

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
        cc = create_cc(pos, feat)

        print(f"""[LOG] Process {rank}: Created combinatorial complex for file {in_filename}""", file=sys.stderr)
        to_pickle(cc, out_dir + out_filename)

    # Finalize MPI
    comm.Barrier()
    MPI.Finalize()

if __name__ == "__main__":
    main()
