from mpi4py import MPI
import numpy as np
import h5py
import random
from itertools import product, combinations
import sys
import os 

from scipy.spatial import Delaunay, distance, KDTree
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.preprocessing import StandardScaler

import networkx as nx
from toponetx.classes.simplicial_complex import SimplicialComplex
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.readwrite.serialization import to_pickle
from toponetx.readwrite.serialization import load_from_pickle

# ---- CONSTANTS ---- #
BOXSIZE = 25e3
DIM = 3
MASS_UNIT = 1e10
Nstar_th = 20 # not used
MASS_CUT = 1e8
ISVOLUME = True
ISDISTANCE = False

# --- HYPERPARAMS --- #
r_link = 0.015
MINCLUSTER = 7 #>10 Found no clusters made in some catalogs. 

# ---- FEATURES ---- #
# NODES:        7 (x, y, z, Mstar, Rstar, Metal, Vmax)
# EDGES:        4 (distance, mid[x,y,z])
# TETRA:        4 (volume, mid[x,y,z])
# CLUSTERS:     8 (avg_volume, std_volume, centroid[x,y,z], std_pos[x,y,z])
# HYPERCLUSTER: 3 (distance, num_galaxies_cluster1, num_galaxies_cluster2)
# ------------------ #

in_dir = "/data2/jylee/topology/IllustrisTNG/data/"
out_dir = "/data2/jylee/topology/IllustrisTNG/combinatorial/cc/"

os.makedirs(out_dir, exist_ok=True)

def normalize(value, isVolume = True):
    global r_link
    if isVolume:
        return value / (r_link**3)
    else:
        return value/ (r_link)

def load_catalog(directory, filename):
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

    # Select only galaxies with more than Nstar_th star particles
    #indexes = np.where(Nstar>Nstar_th)[0]
    indexes = np.where(Mstar>MASS_CUT)[0]
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
    feat = np.hstack((pos, feat))
    return pos, feat

class Edge:
    def __init__(self, nodes, pos=None):
        self.nodes = tuple(sorted(nodes))
        self.pos = np.array([pos[node] for node in self.nodes])
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
        diff = self.pos[0] - self.pos[1]
        for i, coord in enumerate(diff):
            if coord > r_link:
                diff[i] -= 1.0 
            elif coord < -r_link:
                diff[i] += 1.0
        self.distance = np.linalg.norm(diff)
        self.distance = normalize(self.distance, ISDISTANCE)

    def calculate_midpoint(self):
        self.midpoint = np.mean(self.pos, axis=0)
        for i, coord in enumerate(self.midpoint):
            if coord > 1.0:
                self.midpoint[i] -= 1.0
            elif coord < 0.0:
                self.midpoint[i] += 1.0

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
        self.pos = np.array([pos[node] for node in self.nodes])
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
        self.normalized_volume = normalize(self.volume, ISVOLUME)

    def calculate_midpoint(self):
        self.midpoint = np.mean(self.pos, axis=0)

    def add_features(self):
        self.calculate_volume()
        self.calculate_midpoint()
        self.features.append(self.normalized_volume)
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

        self.avg_volume = normalize(np.mean(self.volumes), ISVOLUME)
        self.std_volume = normalize(np.std(self.volumes), ISVOLUME)

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

class Hypercluster:
    def __init__(self, cluster1, cluster2, dist):
        self.cluster1_label = cluster1.label
        self.cluster2_label = cluster2.label
        self.node_set = cluster1.node_set | cluster2.node_set
        
        self.distance = normalize(dist, ISDISTANCE)
        self.num_nodes_cluster1 = np.log10(len(cluster1.node_set)+1)
        self.num_nodes_cluster2 = np.log10(len(cluster2.node_set)+1)

        self.features = [self.distance, self.num_nodes_cluster1, self.num_nodes_cluster2]

    def __repr__(self):
        return f"Hypercluster(cluster1={self.cluster1_label}, cluster2={self.cluster2_label}, features={self.features})"


def create_mst(clusters):
    centroids = np.array([cluster.centroid for cluster in clusters.values()])
    labels = list(clusters.keys())
    dist_matrix = distance.cdist(centroids, centroids, 'euclidean')

    G = nx.Graph()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            G.add_edge(labels[i], labels[j], weight=dist_matrix[i, j])

    mst = nx.minimum_spanning_tree(G)
    return mst


def get_tetrahedra(pos, feat):
    # Generate Delaunay triangulation
    tri = Delaunay(pos)

    # Calculate volumes and store tetrahedra
    tetrahedra = []
    for i, simplex in enumerate(tri.simplices):
        tetra = Tetrahedron(i, simplex, pos)
        tetrahedra.append(tetra)

    tetrahedra.sort(key=lambda x: x.volume)  # Sorting by volume

    # Scale the volumes to make an embedding
    log_volumes = np.array([tetra.log_volume for tetra in tetrahedra])
    volume_scaler = StandardScaler()
    scaled_volumes = volume_scaler.fit_transform(log_volumes.reshape(-1, 1)).flatten()
    
    print("[LOG] GENERATED TETRA", file=sys.stderr)
    return tetrahedra, scaled_volumes

def get_single_tetra_edges(pos, tetra):
    edge_objects = []
    for (i, j) in combinations(range(4), 2):
        edge = tuple(sorted((tetra.nodes[i], tetra.nodes[j])))
        edge_obj = Edge(nodes=edge, pos=pos)
        edge_objects.append(edge_obj)
    return edge_objects

def get_all_tetra_edges(pos, tetrahedra):
    tetra_edge_set = set() # important to make this as a set!!!
    for tetra in tetrahedra:
        tetra_edge_set.update(get_single_tetra_edges(pos, tetra))

    #return tetra_edge_set
    return set()

def get_kdtree_edges(pos, r_link=0.015):
    '''
    Modified from CosmoGraphNet
    arXiv:2204.13713
    https://github.com/PabloVD/CosmoGraphNet/
    '''
    kd_tree = KDTree(pos, leafsize=16, boxsize=1.0001)
    edge_index = kd_tree.query_pairs(r=r_link, output_type="ndarray")

    kdtree_edge_set = set()
    for src, dst in edge_index:
        edge_obj = Edge([src, dst],pos)
        kdtree_edge_set.add(edge_obj)

    #return kdtree_edge_set
    return kdtree_edge_set

def create_cc(in_dir, in_filename):    
    # Read in data
    pos, feat = load_catalog(in_dir, in_filename)

    # 1. Get Tetrahedra
    tetrahedra, scaled_volumes = get_tetrahedra(pos, feat)
    
    # 2. Get edges
    tetra_edge_set = get_all_tetra_edges(pos, tetrahedra)
    kdtree_edge_set = get_kdtree_edges(pos, r_link)
    edge_set = tetra_edge_set | kdtree_edge_set # union
    
    # 3. Clustering on Tetrahedra
    clusters = clustering(tetrahedra, scaled_volumes)

    # 4. Get Hyperclusters using MST
    hyperclusters = {}
    mst = create_mst(clusters)
    for edge in mst.edges(data=True):
        cluster1_label = edge[0]
        cluster2_label = edge[1]
        dist = edge[2]['weight']

        cluster1 = clusters[cluster1_label]
        cluster2 = clusters[cluster2_label]

        hypercluster_label = f"hyper_{cluster1_label}_{cluster2_label}"
        hypercluster = Hypercluster(cluster1, cluster2, dist)
        hyperclusters[hypercluster_label] = hypercluster
    
    ##########################
    NUMPOINTS = pos.shape[0]
    NUMEDGES = len(edge_set)
    NUMTETRA = len(tetrahedra)
    
    # 5. Generate Combinatorial Complex
    print(f"""[LOG] We will select {NUMEDGES} edges and {NUMTETRA} tetra""", file=sys.stderr)
    print(f"""[LOG] Edges from tetra {len(tetra_edge_set)} and KDTree {len(kdtree_edge_set)} with {len(tetra_edge_set & kdtree_edge_set)} edges in common.""", file=sys.stderr)

    cc = CombinatorialComplex()

    ## 5-1 ADD NODES ##
    for node, node_data in zip(list(range(NUMPOINTS)), feat):
        cc.add_cell([node], rank=0)
    
    node_data = {num: feat[num] for num in range(NUMPOINTS)}

    ## 5-2 ADD EDGES ##
    sorted_edge_objects = sorted(list(edge_set), key=lambda e: e.distance if e.features else float('inf'))
    sorted_edges = sorted_edge_objects[:NUMEDGES]

    for edge in sorted_edges:
        cc.add_cell(edge.nodes, rank=1)
    
    edge_data = {edge.nodes: edge.features for edge in sorted_edges}

    ## 5-3 ADD TETRA ##
    sorted_tetra = tetrahedra[:NUMTETRA]

    for tetra in sorted_tetra:
        cc.add_cell(list(tetra.nodes), rank=2)

    tetra_data = {tuple(tetra.nodes): tetra.features for tetra in sorted_tetra}  # Using volume as data

    ## 5-4 ADD Clusters ##
    for cluster in clusters.values():
        cc.add_cell(list(cluster.node_set), rank=3)

    cluster_data = {tuple(cluster.node_set): cluster.features for cluster in clusters.values()}

    # 5-5 ADD HyperCluster ##
    for hypercluster in hyperclusters.values():
        cc.add_cell(list(hypercluster.node_set), rank=4)

    hypercluster_data = {tuple(hypercluster.node_set): hypercluster.features for hypercluster in hyperclusters.values()}

    # ADD CELL ATTRIBUTES
    cc.set_cell_attributes(node_data, name="node_feat")
    cc.set_cell_attributes(edge_data, name="edge_feat")
    cc.set_cell_attributes(tetra_data, name="tetra_feat")
    cc.set_cell_attributes(cluster_data, name="cluster_feat")
    cc.set_cell_attributes(hypercluster_data, name="hypercluster_feat")
    return cc

def remove_subset_clusters(clusters):
    to_remove = set()
    cluster_labels = list(clusters.keys())

    for i in range(len(cluster_labels)):
        for j in range(len(cluster_labels)):
            if i == j:
                continue

            A = clusters[cluster_labels[i]].node_set
            B = clusters[cluster_labels[j]].node_set

            if A & B == A or A & B == B:
                if len(A) <= len(B):
                    to_remove.add(cluster_labels[i])
                else:
                    to_remove.add(cluster_labels[j])

    for label in to_remove:
        del clusters[label]

    print(f"[LOG] Removed {len(to_remove)} subset clusters.")
    return clusters

def clustering(tetrahedra, scaled_volumes):
    global r_link
    embeddings = np.array([tetra.midpoint for i, tetra in enumerate(tetrahedra)])
    db = HDBSCAN(min_samples=MINCLUSTER).fit(embeddings)
    labels = db.labels_

    clusters = {}
    for label in set(labels):
        if label == -1:
            continue
        indices = np.where(labels == label)[0]
        merged_tetra = [tetrahedra[idx] for idx in indices]

        if len(merged_tetra) == 1:
            continue
        else:
            cluster = Cluster(label, merged_tetra, scaled_volumes[indices])
            clusters[label] = cluster

    # Remove subset clusters
    clusters = remove_subset_clusters(clusters)

    print(f"""
    [LOG] We Currently have {len(embeddings)} Tetrahedra.
    [LOG] Generated {len(clusters)} Clusters of Tetrahedra. 
    [LOG] Mean number of nodes per cluster is {np.mean([len(cluster.node_set) for cluster in clusters.values()])}
    [LOG] Max number of nodes per cluster is {np.max([len(cluster.node_set) for cluster in clusters.values()])} and the number is {np.argmax([len(cluster.node_set) for cluster in clusters.values()])}""", file=sys.stderr)

    return clusters

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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

        cc = create_cc(in_dir, in_filename)

        print(f"""[LOG] Process {rank}: Created combinatorial complex for file {in_filename}""", file=sys.stderr)
        to_pickle(cc, out_dir + out_filename)

    comm.Barrier()
    MPI.Finalize()

def main2():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Array to be processed
    array = [34, 35, 36, 37, 38, 39, 40, 41, 57, 58, 59, 60, 61, 62, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 328, 329, 330, 331, 332, 333, 334, 335, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 562, 563, 564, 565, 566, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 755, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 893, 894, 895, 896, 897, 898, 899, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 993, 994, 995, 996, 997, 998, 999]

    total_elements = len(array)
    jobs_per_process = total_elements // size
    extra_jobs = total_elements % size

    # Calculate start and end indices for this process
    if rank < extra_jobs:
        start_idx = rank * (jobs_per_process + 1)
        end_idx = start_idx + jobs_per_process + 1
    else:
        start_idx = rank * jobs_per_process + extra_jobs
        end_idx = start_idx + jobs_per_process

    # Slice the array for this process
    slice_array = array[start_idx:end_idx]
    
    for num in slice_array:
        in_filename = f"data_{num}.hdf5"
        out_filename = f"data_{num}.pickle"

        cc = create_cc(in_dir, in_filename)

        print(f"[LOG] Process {rank}: Created combinatorial complex for file {in_filename}", file=sys.stderr)
        to_pickle(cc, out_dir + out_filename)

    comm.Barrier()
    MPI.Finalize()


if __name__ == "__main__":
    main()

