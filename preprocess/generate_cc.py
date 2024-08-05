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
NUMPOINTS = -1
NUMEDGES = 30000
NUMTETRA = 30000

# FEATURES
# NODES: 5 (x, y, z, Mstar, Rstar)
# EDGES: 4 (distance, mid[x,y,z])
# TETRA: 4 (log_volume, mid[x,y,z])
# CLUSTERS: 8 (avg_volume, std_volume, centroid[x,y,z], std_pos[x,y,z])
# ------------------ #

in_dir = "/data2/jylee/topology/IllustrisTNG/data/"
out_dir = "/data2/jylee/topology/IllustrisTNG/combinatorial/cc_extended_fix/"

def load_catalog(directory, filename):
    f = h5py.File(directory + filename, 'r')
    pos = f['/Subhalo/SubhaloPos'][:] / BOXSIZE
    Mstar = f['/Subhalo/SubhaloMassType'][:, 4]  # Msun/h
    Rstar = f["Subhalo/SubhaloHalfmassRadType"][:, 4]
    feat = np.vstack((Mstar, Rstar)).T
    feat = np.hstack((pos, feat))

    global NUMPOINTS
    NUMPOINTS = pos.shape[0]
    pos = pos[:, :DIM]

    return pos, feat

class Edge:
    def __init__(self, nodes, pos=None):
        self.nodes = tuple(sorted(nodes))
        self.pos = pos  
        self.distance = 0
        self.midpoint = [0,0,0]
        
        self.features = []
        self.add_feature() 

    def add_feature(self):
        self.calculate_distance()
        self.calculate_midpoint()
        self.features.insert(0, self.distance)
        self.features.extend(self.midpoint)

    def calculate_distance(self):
        self.distance = np.linalg.norm(np.array(self.pos[0]) - np.array(self.pos[1]))

    def calculate_midpoint(self):
        self.midpoint = np.mean(np.array(self.pos), axis=0)

    def __repr__(self):
        return f"Edge(nodes={self.nodes}, pos={self.pos}, features={self.features})"

    def __hash__(self):
        return hash(self.nodes)

    def __eq__(self, other):
        return self.nodes == other.nodes

class Tetrahedron:
    def __init__(self, index, nodes, pos):
        self.index = index
        self.nodes = nodes
        self.pos = np.array([pos[node] for node in nodes])
        self.features = []

        self.volume = 0
        self.log_volume = 0
        self.midpoint = [0,0,0]

        self.add_features() 

    def calculate_volume(self):
        mat = np.zeros([4, 4])
        mat[:, :3] = self.pos
        mat[:, 3] = 1.0
        self.volume = np.abs(np.linalg.det(mat)) / 6
        self.log_volume = np.log10(self.volume + 1e-20)

    def calculate_midpoint(self):
        return np.mean(self.pos, axis=0)

    def add_features(self):
        self.calculate_volume()
        self.calculate_midpoint()
        self.features.append(self.log_volume)
        self.features.extend(self.midpoint)
        
    def __repr__(self):
        return f"Tetrahedron(index={self.index}, nodes={self.nodes}, features={self.features})"

class Cluster:
    def __init__(self, label, tetrahedra, scaled_volumes):
        self.label = label
        self.tetrahedra = tetrahedra
        self.scaled_volumes = scaled_volumes
        
        self.volumes = []
        self.node_set = set()
        self.avg_volume = 0
        self.std_volume = 0
        self.centroid = [0,0,0]
        self.std_pos = [0,0,0]
        
        self.features = []
        self.add_features()


    def calculate_volumes(self):
        for tetra in self.tetrahedra:
            self.volumes.append(tetra.volume)
            self.node_set.update(tetra.nodes)

        self.avg_volume = np.log10(np.mean(self.volumes) + 1e-20)
        self.std_volume = np.log10( np.std(self.volumes) + 1e-20)

    def calculate_centroid(self):
        if self.node_set:
            node_positions = []
            for tetra in self.tetrahedra:
                node_positions.extend(tetra.pos)
            
            node_positions = np.array(node_positions)
            if len(node_positions) > 0:
                self.centroid = np.mean(node_positions, axis=0)
                self.std_pos = np.std(node_positions, axis=0)

    def add_features(self):
        self.calculate_volumes()
        self.calculate_centroid()
        
        self.features.append(self.avg_volume)
        self.features.append(self.std_volume)
        self.features.extend(self.centroid)
        self.features.extend(self.std_pos)
        
    def __repr__(self):
        return f"Cluster(label={self.label}, merged_volume={self.merged_volume}, features={self.features})"

def get_tetrahedra(pos, feat):
    # Generate Delaunay triangulation
    tri = Delaunay(pos)

    # Calculate volumes and store tetrahedra
    tetrahedra = []
    for i, simplex in enumerate(tri.simplices):
        tetra = Tetrahedron(i, simplex, pos)
        tetrahedra.append(tetra)

    # Sort tetrahedra by volume (which is the first element in features)
    tetrahedra.sort(key=lambda x: x.volume)  # Sorting by volume

    # Scale the volumes to make an embedding
    log_volumes = np.array([tetra.log_volume for tetra in tetrahedra])
    volume_scaler = StandardScaler()
    scaled_volumes = volume_scaler.fit_transform(log_volumes.reshape(-1, 1)).flatten()
    
    print("[LOG] GENERATED TETRA", file=sys.stderr)
    return tetrahedra, scaled_volumes

def get_single_tetra_edges(tetra):
    edge_objects = []
    for (i, j) in combinations(range(4), 2):
        edge = tuple(sorted((tetra.nodes[i], tetra.nodes[j])))
        pos = [tetra.pos[i], tetra.pos[j]]
        edge_obj = Edge(nodes=edge, pos=pos)
        edge_objects.append(edge_obj)
    return edge_objects

def get_all_tetra_edges(tetrahedra):
    edge_objects = set() # important to make this as a set!!!
    for tetra in tetrahedra:
        edge_objects.update(get_single_tetra_edges(tetra))

    # Sort edges by distance (which is the first element in features)
    sorted_edge_objects = sorted(edge_objects, key=lambda e: e.features[0] if e.features else float('inf'))
    return sorted_edge_objects

def create_cc(pos, feat):
    global NUMEDGES, NUMTETRA
    
    # 1. Get Tetrahedra
    tetrahedra, scaled_volumes = get_tetrahedra(pos, feat)
    
    # 2. Get edges
    edge_objects = get_all_tetra_edges(tetrahedra)
    
    # 3. Clustering on Tetrahedra
    clusters = clustering(tetrahedra, scaled_volumes)
    
    if NUMEDGES == -1 or NUMEDGES > len(edge_objects):
        NUMEDGES = len(edge_objects)
    if NUMTETRA == -1 or NUMTETRA > len(tetrahedra):
        NUMTETRA = len(tetrahedra)
    
    # 4. Generate Combinatorial Complex
    print(f"""[LOG] We will select {NUMEDGES} edges and {NUMTETRA} tetra""", file=sys.stderr)
    cc = CombinatorialComplex()

    ## 4-1 ADD NODES ##
    for node, node_data in zip(list(range(NUMPOINTS)), feat):
        cc.add_cell([node], rank=0)
    
    node_data = {num: feat[num] for num in range(NUMPOINTS)}

    ## 4-2 ADD EDGES ##
    sorted_edges = edge_objects[:NUMEDGES]

    for edge in sorted_edges:
        cc.add_cell(edge.nodes, rank=1)
    
    edge_data = {edge.nodes: edge.features for edge in sorted_edges}

    ## 4-3 ADD TETRA ##
    sorted_tetra = tetrahedra[:NUMTETRA]

    for tetra in sorted_tetra:
        cc.add_cell(list(tetra.nodes), rank=2)

    tetra_data = {tuple(tetra.nodes): tetra.features for tetra in sorted_tetra}  # Using volume as data

    ## 4-4 ADD Clusters ##
    for cluster in clusters.values():
        cc.add_cell(list(cluster.node_set), rank=3)

    cluster_data = {tuple(cluster.node_set): cluster.features for cluster in clusters.values()}

    # ADD CELL ATTRIBUTES
    cc.set_cell_attributes(node_data, name="node_feat")
    cc.set_cell_attributes(edge_data, name="edge_feat")
    cc.set_cell_attributes(tetra_data, name="tetra_feat")
    cc.set_cell_attributes(cluster_data, name="cluster_feat")
    return cc

def clustering(tetrahedra, scaled_volumes):
    # Perform DBSCAN clustering on the midpoints
    embeddings = np.array([np.append(tetra.calculate_midpoint(), scaled_volumes[i]) for i, tetra in enumerate(tetrahedra)])

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
        merged_tetra = [tetrahedra[idx] for idx in indices]
        
        cluster = Cluster(label, merged_tetra, scaled_volumes[indices])
        clusters[label] = cluster

    print(f"""
    [LOG] We Currently have {len(embeddings)} Tetrahedra.
    [LOG] Generated {len(clusters)} Clusters of Tetrahedra. 
    [LOG] Mean number of nodes per cluster is {np.mean([len(cluster.node_set) for cluster in clusters.values()])}
    [LOG] Max number of nodes per cluster is {np.max([len(cluster.node_set) for cluster in clusters.values()])} and the number is {np.argmax([len(cluster.node_set) for cluster in clusters.values()])}""", file=sys.stderr)
    
    return clusters

def main():
    # MPI Initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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

