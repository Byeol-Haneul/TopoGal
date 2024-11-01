import os
from config.hyperparam import HyperparameterTuner, run_optuna_study
from config.machine import *

def tune():
    data_dir_base = DATA_DIR
    checkpoint_dir = RESULT_DIR+"/optimize_small_tnn/"
    label_filename = LABEL_FILENAME
    
    device_num = "0,1,2,3" #not necessary
    n_trials = 100
    only_positions = True

    run_optuna_study(data_dir_base, checkpoint_dir, label_filename, device_num, n_trials, only_positions)

if __name__ == "__main__":
    tune()
