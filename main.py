import logging
import torch
import os, sys
import random
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

from data.load_data import load_tensors, split_data
from model.network import Network
from model.train import train, evaluate, save_checkpoint, load_checkpoint
from data.dataset import CustomDataset, custom_collate_fn

from config.param_config import PARAM_STATS, PARAM_ORDER, normalize_params, denormalize_params

def setup_logger(log_filename):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

def file_cleanup(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.tuning:
        try:
            os.remove(os.path.join(args.checkpoint_dir, 'model_checkpoint.pth'))
            os.remove(os.path.join(args.checkpoint_dir, 'pred.txt'))
            os.remove(os.path.join(args.checkpoint_dir, 'train_losses.csv'))
            os.remove(os.path.join(args.checkpoint_dir, 'val_losses.csv'))
            os.remove(os.path.join(args.checkpoint_dir, 'training.log'))
        except OSError:
            pass 

    log_filename = os.path.join(args.checkpoint_dir, f'training.log')
    setup_logger(log_filename)

    '''
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    '''
    
def gpu_setup(args, local_rank):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    visible_devices = ",".join(str(i) for i in range(torch.cuda.device_count()))
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    torch.cuda.set_device(local_rank)
    args.device = torch.device(f"cuda:{local_rank}")    
    dist.init_process_group(backend="nccl", init_method='env://') 
    print(f"[GPU SETUP] Process {local_rank} set up on device {args.device}", file = sys.stderr)


def fix_random_seed(seed):
    seed = seed if (seed is not None) else 12345
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Random seed fixed to {seed}.")

def implicit_likelihood_loss(output, target):
    num_params = len(target)
    y_out, err_out = output[:,:num_params], output[:,num_params:]
    #loss_mse = torch.mean(torch.sum((y_out - target)**2., axis=1), axis=0)
    #loss_ili = torch.mean(torch.sum(((y_out - target)**2. - err_out**2.)**2., axis=1), axis=0)
    #loss = torch.log(loss_mse) + torch.log(loss_ili)
    loss_mse = torch.mean(torch.sum(torch.abs(y_out - target), axis=1), axis=0)
    loss_ili = torch.mean(torch.sum(torch.abs(torch.abs(y_out - target) - err_out), axis=1), axis=0)
    loss = loss_mse * loss_ili
    return loss

def load_and_prepare_data(num_list, args, local_rank, world_size):
    # Determine the total number of samples
    total_samples = len(num_list)

    # Calculate the number of samples for validation and test
    num_val_samples = int(args.val_size * total_samples)
    num_test_samples = int(args.test_size * total_samples)

    # Calculate the number of training samples
    num_train_samples = total_samples - num_val_samples - num_test_samples

    # Split indices for validation and test
    val_indices = list(range(num_train_samples, num_train_samples + num_val_samples))
    test_indices = list(range(num_train_samples + num_val_samples, total_samples))

    # Prepare data for each rank
    per_process_train_size = num_train_samples // world_size
    start_train_idx = local_rank * per_process_train_size
    end_train_idx = (local_rank + 1) * per_process_train_size

    # Prepare training indices
    train_indices = list(range(start_train_idx, end_train_idx))

    data_dir, label_filename, target_labels, feature_sets = (
        args.data_dir, args.label_filename, args.target_labels, args.feature_sets
    )

    logging.info(f"Loading tensors for {len(num_list)} samples from {data_dir}")
    tensor_dict = load_tensors(
        num_list, data_dir, label_filename, args, target_labels, feature_sets
    )

    logging.info("Normalizing target parameters")
    tensor_dict['y'] = normalize_params(tensor_dict['y'], target_labels)

    # Prepare training data for all ranks
    train_data = {feature: [tensor_dict[feature][i] for i in train_indices] for feature in feature_sets + ['y']}
    train_tuples = list(zip(*[train_data[feature] for feature in feature_sets + ['y']]))
    
    logging.info(f"Created train dataset with {len(train_tuples)} samples")
    train_dataset = CustomDataset(train_tuples, feature_sets + ['y'])

    # Only rank 0 handles validation and test sets
    if local_rank == 0:
        val_data = {feature: [tensor_dict[feature][i] for i in val_indices] for feature in feature_sets + ['y']}
        test_data = {feature: [tensor_dict[feature][i] for i in test_indices] for feature in feature_sets + ['y']}

        val_tuples = list(zip(*[val_data[feature] for feature in feature_sets + ['y']]))
        test_tuples = list(zip(*[test_data[feature] for feature in feature_sets + ['y']]))

        logging.info(f"Created validation dataset with {len(val_tuples)} samples")
        logging.info(f"Created test dataset with {len(test_tuples)} samples")

        val_dataset = CustomDataset(val_tuples, feature_sets + ['y'])
        test_dataset = CustomDataset(test_tuples, feature_sets + ['y'])
    else:
        val_dataset = None
        test_dataset = None

    return train_dataset, val_dataset, test_dataset


def initialize_model(args, local_rank):
    inout_channels = [args.hidden_dim] * len(args.in_channels)
    channels_per_layer = [[args.in_channels, inout_channels]]  # Input layer

    for _ in range(args.num_layers - 1):
        channels_per_layer.append([inout_channels, inout_channels])  # Hidden layers

    logging.info(f"Model architecture: {channels_per_layer}")
    logging.info("Initializing model")
    
    # Define final output layer
    final_output_layer = len(args.target_labels) * 2

    # Initialize the model
    model = Network(args.layerType, channels_per_layer, final_output_layer, args.attention_flag, args.residual_flag)

    # Move the model to the correct device before wrapping in DDP
    model.to(args.device)

    # Only wrap in DDP if there are multiple GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )

    return model


def main(passed_args=None, dataset=None):
    if passed_args is None:
        from config.config import args  # Import only if not passed
    else:
        args = passed_args
    
    # Get local rank and world size from environment variables (usually set by the launcher)
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    ## BASIC SETUP ##
    if local_rank == 0:
        file_cleanup(args)

    fix_random_seed(args.random_seed)
    
    gpu_setup(args, local_rank)

    num_list = [i for i in range(100)]

    if args.tuning:
        train_dataset, val_dataset, test_dataset = dataset
    else:
        train_dataset, val_dataset, test_dataset = load_and_prepare_data(num_list, args, local_rank, world_size)

    #################
    model = initialize_model(args, local_rank) 
    
    # Wrap the model with DDP
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Define loss function and optimizer
    loss_fn = implicit_likelihood_loss
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Define checkpoint path
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model_checkpoint.pth')

    # Training
    logging.info("Starting training")
    best_loss = train(model, train_dataset, val_dataset, test_dataset, loss_fn, opt, args, checkpoint_path, local_rank)
    
    # Evaluation - only from rank 0
    if local_rank == 0:
        logging.info("Starting evaluation")
        evaluate(model, test_dataset, args.device, os.path.join(os.path.dirname(checkpoint_path), "pred.txt"), args.target_labels)
    
    dist.destroy_process_group()

    if local_rank == 0:
        return best_loss

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    main()
