import torch
from argparse import Namespace
from config.machine import BASE_DIR, DATA_DIR, RESULT_DIR

args = Namespace(
    # mode
    tuning=False,
    only_positions=True,

    # Directories
    data_dir=DATA_DIR+"/tensors/",
    checkpoint_dir=RESULT_DIR+"/test_gnn_batching/",
    label_filename=BASE_DIR+"/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt",

    # Model Architecture
    in_channels=[1, 3, 5, 7, 3],
    hidden_dim = 4,
    num_layers=2,
    layerType = "GNN",
    attention_flag = False,
    residual_flag = True,

    # Target Labels
    target_labels = ["Omega0"],

    # Training Hyperparameters
    num_epochs=3000,
    test_interval=10,
    learning_rate=5e-4,#1e-5,
    weight_decay=1e-5,
    batch_size=32,
    drop_prob=0.1,

    # Device
    device_num="0,1",

    # dummies / fixed values 
    device = None, 
    val_size=0.1,
    test_size=0.1,
    random_seed=0,

    # Features & Neighborhood Functions
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
    ],
)
