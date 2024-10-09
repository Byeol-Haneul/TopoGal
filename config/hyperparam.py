import optuna
import optuna_integration
import torch
import torch.distributed as dist

import time
import os, sys
import threading
import json

from argparse import Namespace
from main import main, load_and_prepare_data
from config.machine import MACHINE



class HyperparameterTuner:
    def __init__(self, data_dir, checkpoint_dir, label_filename, device_num, only_positions, local_rank, world_size):
        ## Distributed
        self.local_rank = local_rank
        self.world_size = world_size

        ## Training Config
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.label_filename = label_filename
        self.device_num = device_num
        self.only_positions = only_positions

        # Create a fixed base args for loading data
        self.base_args = self.create_base_args()
        self.dataset = self.load_data()


    def create_base_args(self):
        return Namespace(
            # Mode
            tuning=True,
            only_positions=self.only_positions,

            # Model Architecture
            in_channels=[1, 3, 5, 7, 3],
            attention_flag=False,
            residual_flag=True,

            # Target Labels
            target_labels=["Omega0"],

            # Directories
            data_dir=self.data_dir,
            checkpoint_dir=self.checkpoint_dir,
            label_filename=self.label_filename,

            # Training Hyperparameters
            num_epochs=300,
            test_interval=30,
            batch_size=32,

            # Device
            device_num=self.device_num,
            device=None,

            # Fixed Values
            val_size=0.1,
            test_size=0.1,
            random_seed=12345,

            # Features & Neighborhood Functions
            feature_sets=[
                'x_0', 'x_1', 'x_2', 'x_3', 'x_4',
                'n0_to_0', 'n1_to_1', 'n2_to_2', 'n3_to_3', 'n4_to_4',
                'n0_to_1', 'n0_to_2', 'n0_to_3', 'n0_to_4',
                'n1_to_2', 'n1_to_3', 'n1_to_4',
                'n2_to_3', 'n2_to_4',
                'n3_to_4',
                'euclidean_0_to_0', 'euclidean_1_to_1', 'euclidean_2_to_2', 'euclidean_3_to_3', 'euclidean_4_to_4',
                'euclidean_0_to_1', 'euclidean_0_to_2', 'euclidean_0_to_3', 'euclidean_0_to_4',
                'euclidean_1_to_2', 'euclidean_1_to_3', 'euclidean_1_to_4',
                'euclidean_2_to_3', 'euclidean_2_to_4',
                'euclidean_3_to_4',
                'global_feature'
            ],
        )

    def objective(self, single_trial):
        trial = optuna_integration.TorchDistributedTrial(single_trial)
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
        drop_prob = 0  # Fixed for now
        layer_type = trial.suggest_categorical('layerType', ['Normal'])
        trial_checkpoint_dir = os.path.join(self.checkpoint_dir, f'trial_{trial.number}')
        os.makedirs(trial_checkpoint_dir, exist_ok=True)


        # Create args with the broadcasted hyperparameters
        self.args = self.create_args(trial_checkpoint_dir, hidden_dim, num_layers, learning_rate, weight_decay, layer_type, drop_prob)

        # Train and evaluate
        val_loss = self.train_and_evaluate(self.args)
        return val_loss

    def create_args(self, checkpoint_dir, hidden_dim, num_layers, learning_rate, weight_decay, layer_type, drop_prob):
        # Update the base args with hyperparameters
        args = Namespace(**self.base_args.__dict__)
        args.hidden_dim = hidden_dim
        args.num_layers = num_layers
        args.learning_rate = learning_rate
        args.weight_decay = weight_decay
        args.layerType = layer_type
        args.drop_prob = drop_prob
        args.checkpoint_dir = checkpoint_dir
        return args

    def load_data(self):
        num_list = [i for i in range(1000)]
        return load_and_prepare_data(num_list, self.base_args, self.local_rank, self.world_size)

    def train_and_evaluate(self, args):
        return main(args, self.dataset)
    
    def gpu_setup(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if MACHINE == "HAPPINESS" and self.world_size+1 == torch.cuda.device_count():
            device_num = self.local_rank + 1
        else:
            device_num = self.local_rank
            
        visible_devices = ",".join(str(i) for i in range(torch.cuda.device_count()))
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
        torch.cuda.set_device(device_num)
        self.base_args.device = torch.device(f"cuda:{device_num}")    
        dist.init_process_group(backend="nccl", init_method='env://') 
        print(f"[GPU SETUP] Process {self.local_rank} set up on device {self.base_args.device}", file = sys.stderr)


def run_heartbeat(interval):
    """Heartbeat function to indicate the script is still running."""
    while True:
        print(f"Heartbeat: System running at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(interval)

def run_optuna_study(data_dir, checkpoint_dir, label_filename, device_num, n_trials=50, only_positions=True, heartbeat_interval=1200, study_name="my_study"):
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    tuner = HyperparameterTuner(data_dir, checkpoint_dir, label_filename, device_num, only_positions, local_rank, world_size)
    tuner.gpu_setup()
    study = None
    
    if local_rank == 0:
        # Start the heartbeat thread
        heartbeat_thread = threading.Thread(target=run_heartbeat, args=(heartbeat_interval,))
        heartbeat_thread.daemon = True
        heartbeat_thread.start()

        # Create directory to save the best results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(checkpoint_dir, f"optuna_results_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        # Set up the database URL (for SQLite)
        db_url = f'sqlite:///{os.path.join(checkpoint_dir, "optuna_study.db")}'

        sampler = optuna.samplers.TPESampler(seed=tuner.base_args.random_seed)
        study = optuna.create_study(direction='minimize', sampler=sampler, storage=db_url, study_name=study_name, load_if_exists=True)
        study.optimize(tuner.objective, n_trials=n_trials)
    else:
        for _ in range(n_trials):
            tuner.objective(None)

    if local_rank == 0:
        assert study is not None

        # Save the best hyperparameters
        best_params_path = os.path.join(result_dir, 'best_params.json')
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=4)

        # Save the study results (all trials) to CSV
        trials_df = study.trials_dataframe()
        trials_csv_path = os.path.join(result_dir, 'trials.csv')
        trials_df.to_csv(trials_csv_path, index=False)

        # Plot and save visualizations
        optuna.visualization.matplotlib.plot_optimization_history(study).figure.savefig(os.path.join(result_dir, 'optimization_history.png'))
        optuna.visualization.matplotlib.plot_param_importances(study).figure.savefig(os.path.join(result_dir, 'param_importances.png'))
        optuna.visualization.matplotlib.plot_parallel_coordinate(study).figure.savefig(os.path.join(result_dir, 'parallel_coordinate.png'))

        print(f"Results saved to: {result_dir}")
        print("Best Hyperparameters:", study.best_params)

    dist.destroy_process_group()