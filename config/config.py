import torch
from argparse import Namespace

args = Namespace(
    in_channels=[5, 4, 4, 4],
    intermediate_channels=[64, 64, 64],
    inout_channels=[32, 32, 32],
    num_layers=10,
    final_output_layer=6,
    data_dir="/data2/jylee/topology/IllustrisTNG/combinatorial/tensors_extended/",
    checkpoint_dir="/data2/jylee/topology/IllustrisTNG/combinatorial/checkpoint_extended/",
    label_filename="/data2/jylee/topology/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt",
    num_epochs=100,
    test_interval=10,
    #batch_size=32,
    learning_rate=1e-3,
    val_size=0.15,
    test_size=0.15,
    device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
)
