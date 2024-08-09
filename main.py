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
    loss_mse = torch.sum((y_out - target)**2 , axis=1)
    loss_ili = torch.sum(((y_out - target)**2 - err_out**2)**2, axis=1)
    loss = torch.log(loss_mse) + torch.log(loss_ili)
    return torch.mean(loss)

def load_and_prepare_data(num_list, data_dir, label_filename, test_size, val_size, target_labels):
    logging.info(f"Loading tensors for {len(num_list)} samples from {data_dir}")
    
    y_list, x_0_list, x_1_list, x_2_list, x_3_list, \
    n0_to_0_list, n1_to_1_list, n2_to_2_list, n3_to_3_list, \
    n0_to_1_list, n1_to_2_list, n2_to_3_list, \
    global_feature_list = load_tensors(
        num_list, data_dir, label_filename, target_labels
    )
    
    logging.info("Normalizing target parameters")
    y_list = normalize_params(y_list, target_labels)
    
    logging.info("Splitting data into train, validation, and test sets")
    (y_train, y_val, y_test), (x_0_train, x_0_val, x_0_test), (x_1_train, x_1_val, x_1_test), \
    (x_2_train, x_2_val, x_2_test), (x_3_train, x_3_val, x_3_test), \
    (n0_to_0_train, n0_to_0_val, n0_to_0_test), (n1_to_1_train, n1_to_1_val, n1_to_1_test), \
    (n2_to_2_train, n2_to_2_val, n2_to_2_test), (n3_to_3_train, n3_to_3_val, n3_to_3_test), \
    (n0_to_1_train, n0_to_1_val, n0_to_1_test), (n1_to_2_train, n1_to_2_val, n1_to_2_test), \
    (n2_to_3_train, n2_to_3_val, n2_to_3_test), (global_feature_train, global_feature_val, global_feature_test) = split_data(
        y_list, x_0_list, x_1_list, x_2_list, x_3_list, 
        n0_to_0_list, n1_to_1_list, n2_to_2_list, n3_to_3_list, 
        n0_to_1_list, n1_to_2_list, n2_to_3_list, 
        global_feature_list,
        test_size=test_size, val_size=val_size
    )
    
    train_data = list(zip(y_train, x_0_train, x_1_train, x_2_train, x_3_train, 
                          n0_to_0_train, n1_to_1_train, n2_to_2_train, n3_to_3_train, 
                          n0_to_1_train, n1_to_2_train, n2_to_3_train,
                          global_feature_train))
    val_data = list(zip(y_val, x_0_val, x_1_val, x_2_val, x_3_val, 
                        n0_to_0_val, n1_to_1_val, n2_to_2_val, n3_to_3_val, 
                        n0_to_1_val, n1_to_2_val, n2_to_3_val,
                        global_feature_train))
    test_data = list(zip(y_test, x_0_test, x_1_test, x_2_test, x_3_test, 
                         n0_to_0_test, n1_to_1_test, n2_to_2_test, n3_to_3_test, 
                         n0_to_1_test, n1_to_2_test, n2_to_3_test,
                         global_feature_train))
    
    logging.info(f"Created train dataset with {len(train_data)} samples")
    logging.info(f"Created validation dataset with {len(val_data)} samples")
    logging.info(f"Created test dataset with {len(test_data)} samples")

    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)
    
    return train_dataset, val_dataset, test_dataset

def initialize_model(args):
    channels_per_layer = [
        [args.in_channels, args.intermediate_channels, args.inout_channels],
    ]

    for _ in range(args.num_layers - 1):
        channels_per_layer.append([args.inout_channels, args.intermediate_channels, args.inout_channels])

    logging.info(f"Model architecture: {channels_per_layer}")
    logging.info("Initializing model")
    final_output_layer = len(args.target_labels) * 2
    model = Network(channels_per_layer, final_output_layer).to(args.device) 
    return model

def main(args):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', 
                        handlers=[logging.FileHandler(args.checkpoint_dir + '/' + 'training.log'), logging.StreamHandler()])

    num_list = [i for i in range(1000)]

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.device_num)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare data
    train_dataset, val_dataset, test_dataset = load_and_prepare_data(
        num_list, args.data_dir, args.label_filename, args.test_size, args.val_size, target_labels=args.target_labels
    )
    
    # Define and Initialize model
    model = initialize_model(args) 
    
    # Define loss function and optimizer
    loss_fn = implicit_likelihood_loss
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
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



'''
# We currently do not support batching
logging.info(f"Batch Size: {args.batch_size}")
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)
'''