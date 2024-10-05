import os
from config.hyperparam import HyperparameterTuner, run_optuna_study
from config.machine import BASE_DIR, DATA_DIR, RESULT_DIR

def tune():
    data_dir = DATA_DIR+"/tensors/"
    checkpoint_dir = RESULT_DIR+"/testing/"
    label_filename = BASE_DIR+"/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"
    
    device_num = 1
    n_trials = 30
    only_positions = True

    run_optuna_study(data_dir, checkpoint_dir, label_filename, device_num, n_trials, only_positions)

if __name__ == "__main__":
    tune()
