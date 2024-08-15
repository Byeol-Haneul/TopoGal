import os
from config.hyperparam import HyperparameterTuner, run_optuna_study

def tune():
    data_dir = "/data2/jylee/topology/IllustrisTNG/combinatorial/tensors/"
    checkpoint_dir = "/data2/jylee/topology/IllustrisTNG/combinatorial/checkpoint/"
    label_filename = "/data2/jylee/topology/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"
    
    device_num = 1
    n_trials = 50
    
    run_optuna_study(data_dir, checkpoint_dir, label_filename, device_num, n_trials)

if __name__ == "__main__":
    tune()
