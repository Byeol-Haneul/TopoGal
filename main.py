import torch
import logging
from torch.utils.data import DataLoader
from data.load_data import load_tensors, split_data
from model.network import Network
from model.train import train, evaluate, save_checkpoint, load_checkpoint
from utils.dataset import CustomDataset, custom_collate_fn
from config.config import args
from config.param_config import PARAM_STATS, PARAM_ORDER, normalize_params, denormalize_params
import os
import pandas as pd
import numpy as np

def implicit_likelihood_loss(output, target):
    num_params = len(target)
    y_out, err_out = output[:,:num_params], output[:,num_params:]
    loss_mse = torch.mean(torch.sum((y_out - target)**2., axis=1), axis=0)
    loss_ili = torch.mean(torch.sum(((y_out - target)**2. - err_out**2.)**2., axis=1), axis=0)
    loss = torch.log(loss_mse) + torch.log(loss_ili)
    return loss #torch.mean(loss)

def load_and_prepare_data(num_list, data_dir, label_filename, test_size, val_size, target_labels, feature_sets):
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

    model = Network(args.layerType, channels_per_layer, final_output_layer, args.attention_flag, args.residual_flag).to(args.device)
    return model

def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', 
                        handlers=[logging.FileHandler(args.checkpoint_dir + '/' + 'training.log'), logging.StreamHandler()])

    logging.info(f"Arguments: {vars(args)}")
    
    num_list = [i for i in range(1000)]

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.device_num)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_dataset, val_dataset, test_dataset = load_and_prepare_data(
        num_list, args.data_dir, args.label_filename, args.test_size, args.val_size, target_labels=args.target_labels, feature_sets = args.feature_sets
    )
    
    model = initialize_model(args) 
    
    # Define loss function and optimizer
    loss_fn = implicit_likelihood_loss
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Define checkpoint path
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model_checkpoint.pth')

    # Training
    logging.info("Starting training")
    best_loss = train(model, train_dataset, val_dataset, test_dataset, loss_fn, opt, args, checkpoint_path)
    
    # Evaluation
    logging.info("Starting evaluation")
    evaluate(model, test_dataset, args.device, os.path.join(os.path.dirname(checkpoint_path), "pred.txt"), args.target_labels)

    return best_loss

if __name__ == "__main__":
    main(args)