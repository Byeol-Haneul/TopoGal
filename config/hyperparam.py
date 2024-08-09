import optuna
import torch
import time
import os
from argparse import Namespace
from main import main

class HyperparameterTuner:
    def __init__(self, data_dir, checkpoint_dir, label_filename, device_num):
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.label_filename = label_filename
        self.device_num = device_num

    def objective(self, trial):
        intermediate_channel_size = trial.suggest_int('intermediate_channel_size', 16, 64)
        inout_channel_size = trial.suggest_int('inout_channel_size', 32, 128)
        intermediate_channels = [intermediate_channel_size] * 4
        inout_channels = [inout_channel_size] * 4

        num_layers = trial.suggest_int('num_layers', 2, 5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)

        self.args = self.create_args(
            intermediate_channels=intermediate_channels,
            inout_channels=inout_channels,
            num_layers=num_layers,
            learning_rate=learning_rate
        )

        try:
            val_loss = self.train_and_evaluate(self.args)
        except Exception as e:
            print(f"An error occurred during training: {e}")
            val_loss = float('inf')  # Penalize failed trials

        return val_loss

    def create_args(self, intermediate_channels, inout_channels, num_layers, learning_rate):        
        return Namespace(
            in_channels=[5, 4, 4, 8],
            intermediate_channels=intermediate_channels,
            inout_channels=inout_channels,
            num_layers=num_layers,
            target_labels=["Omega0", "sigma8", "ASN1", "AAGN1", "ASN2", "AAGN2"],
            data_dir=self.data_dir,
            checkpoint_dir=self.checkpoint_dir,
            label_filename=self.label_filename,
            num_epochs=100,
            test_interval=1,
            batch_size=1,
            learning_rate=learning_rate,
            val_size=0.15,
            test_size=0.15,
            device_num=self.device_num,
            device=None
        )

    def train_and_evaluate(self, args):
        return main(args)

def run_optuna_study(data_dir, checkpoint_dir, label_filename, device_num, n_trials=50):
    tuner = HyperparameterTuner(data_dir, checkpoint_dir, label_filename, device_num)
    
    # Create a unique directory to save the best results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(checkpoint_dir, f"optuna_results_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Run the Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(tuner.objective, n_trials=n_trials)
    
    # Save the best parameters to a file
    best_params_path = os.path.join(result_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        optuna.importance.save_study(study, f)
    
    # Save the study results (including all trials) to a CSV file
    trials_df = study.trials_dataframe()
    trials_csv_path = os.path.join(result_dir, 'trials.csv')
    trials_df.to_csv(trials_csv_path, index=False)
    
    print("Best Hyperparameters:", study.best_params)
    print(f"Results saved to: {result_dir}")