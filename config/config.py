import torch
from argparse import Namespace

args = Namespace(
    in_channels=[5, 4, 4, 8],
    intermediate_channels=[32, 32, 32, 32],
    inout_channels=[64, 64, 64, 64],
    num_layers=3,
    target_labels = ["Omega0", "sigma8", "ASN1", "AAGN1", "ASN2", "AAGN2"],
    data_dir="/data2/jylee/topology/IllustrisTNG/combinatorial/tensors/",
    checkpoint_dir="/data2/jylee/topology/IllustrisTNG/combinatorial/checkpoint2/",
    label_filename="/data2/jylee/topology/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt",
    num_epochs=100,
    test_interval=1,
    batch_size=1,
    learning_rate=1e-3,
    val_size=0.15,
    test_size=0.15,
    device_num=2,
    device = None
)
