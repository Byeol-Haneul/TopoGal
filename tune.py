import os
from config.hyperparam import HyperparameterTuner, run_optuna_study
from config.machine import BASE_DIR

def tune():
    data_dir = BASE_DIR+"/IllustrisTNG/combinatorial/tensors/"
    checkpoint_dir = BASE_DIR+"/IllustrisTNG/combinatorial/results/optimize_gnn/"
    label_filename = BASE_DIR+"/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"
    
    device_num = 0
    n_trials = 70
    only_positions = True

    run_optuna_study(data_dir, checkpoint_dir, label_filename, device_num, n_trials, only_positions)

if __name__ == "__main__":
    tune()
