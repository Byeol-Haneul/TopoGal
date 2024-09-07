import os
from config.hyperparam import HyperparameterTuner, run_optuna_study

def tune():
    data_dir = "/data2/jylee/topology/IllustrisTNG/combinatorial/tensors_test/"
    checkpoint_dir = "/data2/jylee/topology/IllustrisTNG/combinatorial/optimize_test/"
    label_filename = "/data2/jylee/topology/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"
    
    device_num = 2
    n_trials = 50
    only_positions = True
    
    run_optuna_study(data_dir, checkpoint_dir, label_filename, device_num, n_trials, only_positions)

if __name__ == "__main__":
    tune()
