import torch
from argparse import Namespace

args = Namespace(
    in_channels=[3, 4, 4, 8, 3],
    intermediate_channels=[64, 64, 64, 64, 64],
    inout_channels=[64, 64, 64, 64, 64],
    num_layers=1,
    target_labels = ["Omega0"], #["Omega0", "sigma8", "ASN1", "AAGN1", "ASN2", "AAGN2"]
    data_dir="/data2/jylee/topology/IllustrisTNG/combinatorial/tensors_low/",
    checkpoint_dir="/data2/jylee/topology/IllustrisTNG/combinatorial/checkpoint8_low_onlyPos_Om_Hier",
    label_filename="/data2/jylee/topology/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt",
    num_epochs=100,
    test_interval=5,
    batch_size=1,
    learning_rate=1e-6,
    weight_decay=1e-7,
    val_size=0.15,
    test_size=0.15,
    device_num=2,
    device = None,
    layerType = "Master",
    
    feature_sets = [
        'x_0', 'x_1', 'x_2', 'x_3', 'x_4',

        'n0_to_0', 'n1_to_1', 'n2_to_2', 'n3_to_3', 'n4_to_4',
        'n0_to_1', 'n0_to_2', 'n0_to_3', 'n0_to_4',
        'n1_to_2', 'n1_to_3', 'n1_to_4',
        'n2_to_3', 'n2_to_4',
        'n3_to_4',

        'global_feature'
    ]
)
