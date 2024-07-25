import torch
from argparse import Namespace

args = Namespace(
    in_channels=[5, 1, 1, 1],
    intermediate_channels=[32, 32, 32],
    inout_channels=[16, 16, 16],
    num_layers=4,
    final_output_layer=6,
    data_dir="/data2/jylee/topology/IllustrisTNG/combinatorial/tensors/",
    checkpoint_dir="/data2/jylee/topology/IllustrisTNG/combinatorial/checkpoint/",
    label_filename="/data2/jylee/topology/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt",
    num_epochs=100,
    test_interval=10,
    #batch_size=32,
    learning_rate=0.001,
    val_size=0.15,
    test_size=0.15,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
)
