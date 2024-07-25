import torch
import logging
from torch.utils.data import DataLoader
from data.load_data import load_tensors, split_data
from model.network import Network
from model.train import train, evaluate, save_checkpoint, load_checkpoint
from utils.dataset import CustomDataset
from config.config import args
import os
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()])

def load_and_prepare_data(num_list, data_dir, label_filename, test_size, val_size):
    logging.info(f"Loading tensors for {len(num_list)} samples from {data_dir}")
    
    y_list, x_0_list, x_1_list, x_2_list, x_3_list, \
    n0_to_0_list, n1_to_1_list, n2_to_2_list, n0_to_1_list, n1_to_2_list = load_tensors(
        num_list, data_dir, label_filename
    )
    
    logging.info("Splitting data into train, validation, and test sets")
    (y_train, y_val, y_test), (x_0_train, x_0_val, x_0_test), (x_1_train, x_1_val, x_1_test), \
    (x_2_train, x_2_val, x_2_test), (x_3_train, x_3_val, x_3_test), (n0_to_0_train, n0_to_0_val, n0_to_0_test), \
    (n1_to_1_train, n1_to_1_val, n1_to_1_test), (n2_to_2_train, n2_to_2_val, n2_to_2_test), \
    (n0_to_1_train, n0_to_1_val, n0_to_1_test), (n1_to_2_train, n1_to_2_val, n1_to_2_test) = split_data(
        y_list, x_0_list, x_1_list, x_2_list, x_3_list, n0_to_0_list, n1_to_1_list, n2_to_2_list, 
        n0_to_1_list, n1_to_2_list, test_size=test_size, val_size=val_size
    )
    
    train_data = list(zip(y_train, x_0_train, x_1_train, x_2_train, x_3_train, 
                          n0_to_0_train, n1_to_1_train, n2_to_2_train, n0_to_1_train, n1_to_2_train))
    val_data = list(zip(y_val, x_0_val, x_1_val, x_2_val, x_3_val, 
                        n0_to_0_val, n1_to_1_val, n2_to_2_val, n0_to_1_val, n1_to_2_val))
    test_data = list(zip(y_test, x_0_test, x_1_test, x_2_test, x_3_test, 
                         n0_to_0_test, n1_to_1_test, n2_to_2_test, n0_to_1_test, n1_to_2_test))
    
    logging.info(f"Created train dataset with {len(train_data)} samples")
    logging.info(f"Created validation dataset with {len(val_data)} samples")
    logging.info(f"Created test dataset with {len(test_data)} samples")
    
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)
    
    return train_dataset, val_dataset, test_dataset

def initialize_model(channels_per_layer, final_output_layer, device):
    logging.info("Initializing model")
    model = Network(channels_per_layer, final_output_layer).to(device)
    return model

def main(args):
    num_list = [i for i in range(1000)]
    
    # Load and prepare data
    train_dataset, val_dataset, test_dataset = load_and_prepare_data(
        num_list, args.data_dir, args.label_filename, args.test_size, args.val_size
    )
    
    # Define the channels per layer
    channels_per_layer = [
        [args.in_channels[:3], args.intermediate_channels, args.inout_channels],
        [args.inout_channels, args.intermediate_channels, args.inout_channels],
        [args.inout_channels, args.intermediate_channels, args.inout_channels]
    ]

    logging.info(f"Model architecture: {channels_per_layer}")
    
    # Initialize model
    model = initialize_model(channels_per_layer, args.final_output_layer, args.device)
    
    # Define loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Define checkpoint path
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model_checkpoint.pth')

    # Train the model
    logging.info("Starting training")
    train(model, train_dataset, val_dataset, test_dataset, loss_fn, opt, args.num_epochs, args.test_interval, args.device, checkpoint_path)
    
    # Evaluate the model
    logging.info("Starting evaluation")
    pred_filename = os.path.join(args.checkpoint_dir, 'pred.txt')
    evaluate(model, test_dataset, args.device, pred_filename)

if __name__ == "__main__":
    main(args)
