import torch
from argparse import Namespace

args = Namespace(
    in_channels=[5, 4, 4, 8],
    intermediate_channels=[32, 32, 32, 32],
    inout_channels=[64, 64, 64, 64],
    num_layers=3,
    final_output_layer=12,  # doubled for std_devs. 
    data_dir="/data2/jylee/topology/IllustrisTNG/combinatorial/tensors/",
    checkpoint_dir="/data2/jylee/topology/IllustrisTNG/combinatorial/checkpoint/",
    label_filename="/data2/jylee/topology/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt",
    num_epochs=100,
    test_interval=1,
    batch_size=1,
    learning_rate=1e-3,
    val_size=0.15,
    test_size=0.15,
    device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
)
