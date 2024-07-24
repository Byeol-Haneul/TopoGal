import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os

def save_checkpoint(model, optimizer, epoch, loss, path):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(state, path)
    logging.info(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, path, device):
    if os.path.isfile(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch = state['epoch']
        loss = state['loss']
        logging.info(f"Checkpoint loaded from {path}")
        return model, optimizer, epoch, loss
    else:
        logging.error(f"No checkpoint found at {path}")
        return model, optimizer, 0, float('inf')

def train(model, train_dataset, test_dataset, loss_fn, opt, num_epochs, test_interval, device, checkpoint_path):
    start_epoch = 1
    best_loss = float('inf')
    
    # Load checkpoint if exists
    if os.path.isfile(checkpoint_path):
        model, opt, start_epoch, best_loss = load_checkpoint(model, opt, checkpoint_path, device)

    for epoch_i in range(start_epoch, num_epochs + 1):
        epoch_loss = []
        model.train()
        
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            x_0 = sample['x_0'].float().to(device)
            x_1 = sample['x_1'].float().to(device)
            x_2 = sample['x_2'].float().to(device)
            x_3 = sample['x_3'].float().to(device)
            n0_to_0 = sample['n0_to_0'].float().to(device)
            n1_to_1 = sample['n1_to_1'].float().to(device)
            n2_to_2 = sample['n2_to_2'].float().to(device)
            n0_to_1 = sample['n0_to_1'].float().to(device)
            n1_to_2 = sample['n1_to_2'].float().to(device)
            y = sample['y'].float().to(device)
            
            opt.zero_grad()
            y_hat = model(x_0, x_1, x_2, n0_to_0, n1_to_1, n2_to_2, n0_to_1, n1_to_2)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())
            logging.debug(f"Iteration: {i}, Loss: {loss.item():.4f}")
        
        avg_loss = np.mean(epoch_loss)
        logging.info(f"Epoch: {epoch_i}, Loss: {avg_loss:.4f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, opt, epoch_i, best_loss, checkpoint_path)
        
        if epoch_i % test_interval == 0:
            evaluate(model, test_dataset, device)

def evaluate(model, test_dataset, device):
    model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            x_0 = sample['x_0'].float().to(device)
            x_1 = sample['x_1'].float().to(device)
            x_2 = sample['x_2'].float().to(device)
            x_3 = sample['x_3'].float().to(device)
            n0_to_0 = sample['n0_to_0'].float().to(device)
            n1_to_1 = sample['n1_to_1'].float().to(device)
            n2_to_2 = sample['n2_to_2'].float().to(device)
            n0_to_1 = sample['n0_to_1'].float().to(device)
            n1_to_2 = sample['n1_to_2'].float().to(device)
            y = sample['y'].float().to(device)
            
            y_hat = model(x_0, x_1, x_2, n0_to_0, n1_to_1, n2_to_2, n0_to_1, n1_to_2)
            logging.debug(f"y: {y}, y_hat: {y_hat}")
