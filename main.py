import logging
import torch
import os
import random
import pandas as pd
import numpy as np
import sys


from torch.utils.data import DataLoader

from data.load_data import load_tensors, split_data
from model.network import Network
from model.train import train, evaluate, save_checkpoint, load_checkpoint
from data.dataset import CustomDataset, collate_fn

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

    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")

def gpu_setup(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.device_num)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

def fix_random_seed(seed):
    seed = seed if (seed is not None) else 12345
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Random seed fixed to {seed}.")

def implicit_likelihood_loss(output, target):
    num_params = target.shape[-1]
    y_out, err_out = output[:,:num_params], output[:,num_params:]
    #loss_ili = torch.mean(torch.sum(((y_out - target)**2. - err_out**2.)**2., axis=1), axis=0)
    #loss = torch.log(loss_mse) + torch.log(loss_ili)
    loss_mse = torch.mean(torch.sum(torch.abs(y_out - target), axis=1), axis=0)
    loss_ili = torch.mean(torch.sum(torch.abs(torch.abs(y_out - target) - err_out), axis=1), axis=0)
    loss = loss_mse * loss_ili
    return loss

def load_and_prepare_data(num_list, args):
    data_dir, label_filename, test_size, val_size, target_labels, feature_sets = (
        args.data_dir, args.label_filename, args.test_size, args.val_size, args.target_labels, args.feature_sets
    )

    logging.info(f"Loading tensors for {len(num_list)} samples from {data_dir}")
    tensor_dict = load_tensors(
        num_list, data_dir, label_filename, args, target_labels, feature_sets
    )
    
    logging.info("Normalizing target parameters")
    tensor_dict['y'] = normalize_params(tensor_dict['y'], target_labels)
    
    logging.info("Splitting data into train, validation, and test sets")
    train_data = {}
    val_data = {}
    test_data = {}
    
    train_idx, val_idx, test_idx = split_data(
        num_list, test_size=test_size, val_size=val_size
    )

    for feature in feature_sets + ['y']:
        feature_data = tensor_dict[feature]
        train_data[feature] = [feature_data[i] for i in train_idx]
        val_data[feature] = [feature_data[i] for i in val_idx]
        test_data[feature] = [feature_data[i] for i in test_idx]
    
    train_tuples = list(zip(*[train_data[feature] for feature in feature_sets + ['y']]))
    val_tuples = list(zip(*[val_data[feature] for feature in feature_sets + ['y']]))
    test_tuples = list(zip(*[test_data[feature] for feature in feature_sets + ['y']]))

    logging.info(f"Created train dataset with {len(train_tuples)} samples")
    logging.info(f"Created validation dataset with {len(val_tuples)} samples")
    logging.info(f"Created test dataset with {len(test_tuples)} samples")
    
    train_dataset = CustomDataset(train_tuples, feature_sets + ['y'])
    val_dataset = CustomDataset(val_tuples, feature_sets + ['y'])
    test_dataset = CustomDataset(test_tuples, feature_sets + ['y'])
    
    return train_dataset, val_dataset, test_dataset

def initialize_model(args):
    inout_channels = [args.hidden_dim] * len(args.in_channels)
    channels_per_layer = [
        [args.in_channels, inout_channels],
    ]

    for _ in range(args.num_layers - 1):
        channels_per_layer.append([inout_channels, inout_channels])

    logging.info(f"Model architecture: {channels_per_layer}")
    logging.info("Initializing model")
    final_output_layer = len(args.target_labels) * 2

    model = Network(args.layerType, channels_per_layer, final_output_layer, args.attention_flag, args.residual_flag)
    
    # Wrap the model for multi-GPU
    '''
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    '''
    return model.to(args.device)


def main(passed_args=None, dataset=None):
    # Use external args if provided, otherwise use the default config.args
    if passed_args is None:
        from config.config import args  # Import only if not passed
    else:
        args = passed_args

    ## BASIC SETUP ##
    file_cleanup(args)
    fix_random_seed(args.random_seed)
    gpu_setup(args)


    num_list = [i for i in range(1000)]

    if args.tuning:
        train_dataset, val_dataset, test_dataset = dataset
    else:
        train_dataset, val_dataset, test_dataset = load_and_prepare_data(num_list, args)

    #################
    model = initialize_model(args) 
    
    # Define loss function and optimizer
    loss_fn = implicit_likelihood_loss
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    #opt = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)

    # Define checkpoint path
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model_checkpoint.pth')


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Training
    logging.info("Starting training")
    best_loss = train(model, train_loader, val_loader, test_loader, loss_fn, opt, args, checkpoint_path)
    
    # Evaluation
    logging.info("Starting evaluation")
    evaluate(model, test_dataset, args.device, os.path.join(os.path.dirname(checkpoint_path), "pred.txt"), args.target_labels)

    return best_loss

if __name__ == "__main__":
    main()
