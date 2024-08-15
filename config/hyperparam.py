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
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
        num_layers = trial.suggest_int('num_layers', 1, 5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-7, 1e-4)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-4)
        layer_type = trial.suggest_categorical('layerType', ['GNN', 'Normal', 'Master'])

        self.args = self.create_args(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            layer_type=layer_type
        )

        try:
            val_loss = self.train_and_evaluate(self.args)
        except Exception as e:
            print(f"An error occurred during training: {e}")
            val_loss = float('inf')  # Penalize failed trials

        return val_loss

    def create_args(self, hidden_dim, num_layers, learning_rate, weight_decay, layer_type):        
        return Namespace(
            # Model Architecture
            in_channels=[7, 4, 4, 8, 3],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            layerType=layer_type,  # Add layer type here
            attention_flag=False,
            residual_flag=True,

            # Target Labels
            target_labels=["Omega0", "sigma8", "ASN1", "AAGN1", "ASN2", "AAGN2"],

            # Directories
            data_dir=self.data_dir,
            checkpoint_dir=self.checkpoint_dir,
            label_filename=self.label_filename,

            # Training Hyperparameters
            num_epochs=100,
            test_interval=1,
            learning_rate=learning_rate,
            weight_decay=weight_decay,

            # Device
            device_num=self.device_num,
            device=torch.device(f"cuda:{self.device_num}" if torch.cuda.is_available() else "cpu"),

            # Fixed Values
            batch_size=1,
            val_size=0.15,
            test_size=0.15,

            # Features & Neighborhood Functions
            feature_sets=[
                'x_0', 'x_1', 'x_2', 'x_3', 'x_4',
                'n0_to_0', 'n1_to_1', 'n2_to_2', 'n3_to_3', 'n4_to_4',
                'n0_to_1', 'n0_to_2', 'n0_to_3', 'n0_to_4',
                'n1_to_2', 'n1_to_3', 'n1_to_4',
                'n2_to_3', 'n2_to_4',
                'n3_to_4',
                'global_feature'
            ],
        )

    def train_and_evaluate(self, args):
        return main(args)

def run_optuna_study(data_dir, checkpoint_dir, label_filename, device_num, n_trials=50):
    tuner = HyperparameterTuner(data_dir, checkpoint_dir, label_filename, device_num)
    
    # Create directory to save the best results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(checkpoint_dir, f"optuna_results_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
     
    # Run the Optuna study
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(tuner.objective, n_trials=n_trials)
    
    best_params_path = os.path.join(result_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        optuna.importance.save_study(study, f)
    
    # Save the study results (including all trials) to a CSV file
    trials_df = study.trials_dataframe()
    trials_csv_path = os.path.join(result_dir, 'trials.csv')
    trials_df.to_csv(trials_csv_path, index=False)
    
    # Print the best hyperparameters
    print("Best Hyperparameters:", study.best_params)
    print(f"Results saved to: {result_dir}")
    
    # Plot and save visualizations
    plot_optimization_history_path = os.path.join(result_dir, 'optimization_history.png')
    plot_param_importances_path = os.path.join(result_dir, 'param_importances.png')
    plot_parallel_coordinate_path = os.path.join(result_dir, 'parallel_coordinate.png')
    
    optuna.visualization.matplotlib.plot_optimization_history(study).figure.savefig(plot_optimization_history_path)
    optuna.visualization.matplotlib.plot_param_importances(study).figure.savefig(plot_param_importances_path)
    optuna.visualization.matplotlib.plot_parallel_coordinate(study).figure.savefig(plot_parallel_coordinate_path)
    
    print(f"Optimization history saved to: {plot_optimization_history_path}")
    print(f"Parameter importances saved to: {plot_param_importances_path}")
    print(f"Parallel coordinate plot saved to: {plot_parallel_coordinate_path}")