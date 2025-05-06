import os
import shutil
import h5py
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import logging
from multiprocessing import Manager
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import optuna, optuna_integration

from config.machine import *  # adjust as needed
from config.param_config import normalize_params, denormalize_params
from main import fix_random_seed

# Setup global logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ---------- Dataset ----------
class TPCFDataset(Dataset):
    def __init__(self, input_file, label_file, target_labels, indices):
        super().__init__()
        with h5py.File(input_file, 'r') as f_in:
            self.tpcf = torch.tensor(f_in['tpcf'][:], dtype=torch.float32)
            self.inputs = self.tpcf
            self.indices = indices

        with h5py.File(label_file, 'r') as f_lab:
            labels = [f_lab['params'][param][:] for param in target_labels]
            labels = np.stack(labels, axis=-1) 
            labels = torch.tensor(labels, dtype=torch.float32)
            self.labels = normalize_params(labels, target_labels)

    def __len__(self):
        return self.tpcf.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[self.indices[idx]]

# ---------- Model ----------
class TPCF_NN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout_rate)]
            input_dim = h
        layers.append(nn.Linear(input_dim, 2))  # Omega_m, sigma_8
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------- Training Function ----------
def train(rank, world_size, args):
    # load train and val
    train_ds = TPCFDataset(args['train_input'], args['train_label'], args['target_labels'], args["train_indices"])
    val_ds   = TPCFDataset(args['val_input'],   args['val_label'],   args['target_labels'], args["val_indices"])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, world_size, rank)
    val_sampler   = torch.utils.data.distributed.DistributedSampler(val_ds,   world_size, rank)

    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], sampler=train_sampler)
    val_loader   = DataLoader(val_ds,   batch_size=args['batch_size'], sampler=val_sampler)

    model = TPCF_NN(train_ds.inputs.shape[1], args['hidden_dims'], args['dropout']).to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_model_state = None

    for epoch in range(args['epochs']):
        model.train()
        train_sampler.set_epoch(epoch)
        running_train = 0.0
        for x, y in train_loader:
            x, y = x.to(rank), y.to(rank)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_train += loss.item() * x.size(0)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(rank), y.to(rank)
                running_val += criterion(model(x), y).item() * x.size(0)

        if rank == 0:
            avg_train = running_train / len(train_loader.dataset)
            avg_val   = running_val   / len(val_loader.dataset)
            logging.info(f"Epoch {epoch}: Train {avg_train:.4f}, Val {avg_val:.4f}")

            # Save best model so far
            if avg_val < best_val:
                best_val = avg_val
                best_model_state = model.module.state_dict()

    # After training, restore the best model
    if rank == 0 and best_model_state is not None:
        model.module.load_state_dict(best_model_state)

    return model.module if isinstance(model, DDP) else model, best_val

# ---------- Evaluation Function ----------
def evaluate(model, pred_filename, args):
    ds = TPCFDataset(args['test_input'], args['val_label'], args['target_labels'], args["val_indices"])

    target_labels = args['target_labels']
    loader = DataLoader(ds, batch_size=args['batch_size'], shuffle=False)    
    model.eval()
    preds, reals = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(0)
            y_hat = model(x)
            preds.extend(y_hat.cpu().numpy())
            reals.extend(y.cpu().numpy())

    denormalized_real_values = denormalize_params(np.array(reals), target_labels)
    denormalized_predictions = denormalize_params(np.array(preds), target_labels)
           
    pred_df = pd.DataFrame({
        "real": list(denormalized_real_values),
        "pred": list(denormalized_predictions)
    })
    pred_df.to_csv(pred_filename, index=False)
    logging.info(f"Predictions saved to {pred_filename}")

# ---------- Wrapper for DDP spawn ----------
def train_wrapper(args, trial_num):
    global_rank = int(os.environ['RANK'])    
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    splits = np.load(BASE_DIR + "splits.npz")
    args["train_indices"] = splits["train"]
    args["val_indices"] = splits["val"]
    args["test_indices"] = splits["test"]

    # Setup trial directory
    trial_dir = os.path.join(RESULT_DIR, f'trial_{trial_num}')
    os.makedirs(trial_dir, exist_ok=True)
    log_path = os.path.join(trial_dir, 'training.log')
    fh = logging.FileHandler(log_path)
    logging.getLogger().addHandler(fh)

    logging.info(f"Starting trial {trial_num} on rank {rank}")

    if rank == 0:
        logging.info(f"Trial parameters: {args}")

    # Copy the script to the trial directory
    model, val_loss = train(rank, world_size, args)
    
    if rank == 0:
        # save current and best models
        curr_path = os.path.join(trial_dir, 'current_model.pt')
        best_path = os.path.join(trial_dir, 'best_model.pt')
        torch.save(model.state_dict(), curr_path)
        torch.save(model.state_dict(), best_path)
        logging.info(f"Saved models at {curr_path} and {best_path}")
        # evaluate and save predictions
        pred_filename = os.path.join(trial_dir, 'pred.txt')
        evaluate(model, pred_filename, args)
    
    val_loss = torch.tensor(val_loss, device=rank)
    dist.broadcast(val_loss, src=0)
    return val_loss.item()

# ---------- Optuna Objective ----------
def objective(trial):
    trial = optuna_integration.TorchDistributedTrial(trial)
    args = {
        'train_input': HDF5_DATA_FILE.strip(".hdf5")+"_train.hdf5",
        'val_input':   HDF5_DATA_FILE.strip(".hdf5")+"_val.hdf5",
        'test_input':  HDF5_DATA_FILE.strip(".hdf5")+"_test.hdf5",
        'train_label': LABEL_FILENAME.strip(".hdf5")+"_train.hdf5",
        'val_label':   LABEL_FILENAME.strip(".hdf5")+"_val.hdf5",
        'test_label':  LABEL_FILENAME.strip(".hdf5")+"_test.hdf5",
        'target_labels': ['Omega_m', 'sigma_8'],
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'epochs': 300,
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'hidden_dims': [
            trial.suggest_int('h1', 64, 128),
            trial.suggest_int('h2', 64, 128),
            trial.suggest_int('h3', 16, 64),
        ],
        'random_seed': 12345,
    }

    fix_random_seed(args["random_seed"])
    val_loss = train_wrapper(args, trial.number)
    return val_loss

# ---------- Main ----------
if __name__ == '__main__':
    global_rank = int(os.environ['RANK'])    
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(backend="nccl", init_method='env://') 

    n_trials = 100

    if global_rank == 0:
        os.makedirs(RESULT_DIR, exist_ok=True)
        db_url = f'sqlite:///{os.path.join(RESULT_DIR, "optuna_study.db")}'

        sampler = optuna.samplers.TPESampler(seed=12345)
        study = optuna.create_study(direction='minimize', sampler=sampler, storage=db_url, load_if_exists=False)
        study.optimize(objective, n_trials=n_trials)
    else:
        for _ in range(n_trials):
            objective(None)

    print('Best Trial Params:', study.best_trial.params)
    dist.destroy_process_group()
