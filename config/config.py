import torch
from argparse import Namespace

args = Namespace(
    # Directories
    data_dir="/data2/jylee/topology/IllustrisTNG/combinatorial/tensors/",
    checkpoint_dir="/data2/jylee/topology/IllustrisTNG/combinatorial/checkpoint19",
    label_filename="/data2/jylee/topology/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt",


    # Model Architecture
    in_channels=[7, 4, 4, 8, 3],
    intermediate_channels=[64, 64, 64, 64, 64],
    inout_channels=[64, 64, 64, 64, 64],
    num_layers=2,
    layerType = "GNN",
    attention_flag = False,
    residual_flag = True,

    # Target Labels
    target_labels = ["Omega0", "sigma8", "ASN1", "AAGN1", "ASN2", "AAGN2"],

    # Training Hyperparameters
    num_epochs=100,
    test_interval=5,
    learning_rate=1e-5,#1e-5,
    weight_decay=1e-5,

    # Device
    device_num=1,

    # dummies / fixed values 
    device = None, 
    batch_size=1,
    val_size=0.15,
    test_size=0.15,

    # Features & Neighborhood Functions
    feature_sets = [
        'x_0', 'x_1', 'x_2', 'x_3', 'x_4',

        'n0_to_0', 'n1_to_1', 'n2_to_2', 'n3_to_3', 'n4_to_4',
        'n0_to_1', 'n0_to_2', 'n0_to_3', 'n0_to_4',
        'n1_to_2', 'n1_to_3', 'n1_to_4',
        'n2_to_3', 'n2_to_4',
        'n3_to_4',

        'global_feature'
    ],
)
