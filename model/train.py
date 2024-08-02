import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os
import pandas as pd

from config.param_config import PARAM_STATS, PARAM_ORDER, denormalize_params


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

def train(model, train_dataset, val_dataset, test_dataset, loss_fn, opt, num_epochs, test_interval, device, checkpoint_path):
    start_epoch = 1
    best_loss = float('inf')
    
    # Load checkpoint if exists
    if os.path.isfile(checkpoint_path):
        model, opt, start_epoch, best_loss = load_checkpoint(model, opt, checkpoint_path, device)
    
    train_losses = []
    val_losses = []

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
            n3_to_3 = sample['n3_to_3'].float().to(device)
            n0_to_1 = sample['n0_to_1'].float().to(device)
            n1_to_2 = sample['n1_to_2'].float().to(device)
            n2_to_3 = sample['n2_to_3'].float().to(device)
            y = sample['y'].float().to(device)
            
            opt.zero_grad()
            y_hat = model(x_0, x_1, x_2, x_3, n0_to_0, n1_to_1, n2_to_2, n3_to_3, n0_to_1, n1_to_2, n2_to_3)

            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())
            logging.debug(f"Epoch: {epoch_i}, Train Iteration: {i}, Loss: {loss.item():.4f}")
        
        avg_train_loss = np.mean(epoch_loss)
        logging.info(f"Epoch: {epoch_i}, Train Loss: {avg_train_loss:.4f}")
        train_losses.append(avg_train_loss)
        
        # Validate the model
        val_loss = validate(model, val_dataset, loss_fn, device, epoch_i)
        logging.info(f"Epoch: {epoch_i}, Validation Loss: {val_loss:.4f}")
        val_losses.append(val_loss)
        
        # Save checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, opt, epoch_i, best_loss, checkpoint_path)
        
        # Save the train and validation losses after each epoch
        loss_dir = os.path.dirname(checkpoint_path)
        pd.DataFrame({"train_loss": train_losses}).to_csv(os.path.join(loss_dir, "train_losses.csv"), index=False)
        pd.DataFrame({"val_loss": val_losses}).to_csv(os.path.join(loss_dir, "val_losses.csv"), index=False)
        
        if epoch_i % test_interval == 0:
            logging.info(f"Starting evaluation for epoch {epoch_i}")
            evaluate(model, test_dataset, device, os.path.join(os.path.dirname(checkpoint_path), "pred.txt"))

def validate(model, val_dataset, loss_fn, device, epoch_i):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            x_0 = sample['x_0'].float().to(device)
            x_1 = sample['x_1'].float().to(device)
            x_2 = sample['x_2'].float().to(device)
            x_3 = sample['x_3'].float().to(device)
            n0_to_0 = sample['n0_to_0'].float().to(device)
            n1_to_1 = sample['n1_to_1'].float().to(device)
            n2_to_2 = sample['n2_to_2'].float().to(device)
            n3_to_3 = sample['n3_to_3'].float().to(device)
            n0_to_1 = sample['n0_to_1'].float().to(device)
            n1_to_2 = sample['n1_to_2'].float().to(device)
            n2_to_3 = sample['n2_to_3'].float().to(device)
            y = sample['y'].float().to(device)
            
            y_hat = model(x_0, x_1, x_2, x_3, n0_to_0, n1_to_1, n2_to_2, n3_to_3, n0_to_1, n1_to_2, n2_to_3)
            loss = loss_fn(y_hat, y)
            val_loss.append(loss.item())
            logging.debug(f"Epoch: {epoch_i}, Validation Iteration: {i}, Loss: {loss.item():.4f}")
    
    return np.mean(val_loss)

def evaluate(model, test_dataset, device, pred_filename):
    model.eval()
    predictions = []
    real_values = []
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
            n3_to_3 = sample['n3_to_3'].float().to(device)
            n0_to_1 = sample['n0_to_1'].float().to(device)
            n1_to_2 = sample['n1_to_2'].float().to(device)
            n2_to_3 = sample['n2_to_3'].float().to(device)
            y = sample['y'].float().to(device)

            y_hat = model(x_0, x_1, x_2, x_3, n0_to_0, n1_to_1, n2_to_2, n3_to_3, n0_to_1, n1_to_2, n2_to_3)
            predictions.extend(y_hat.cpu().numpy())
            real_values.append(y.cpu().numpy())
            logging.debug(f"Test Iteration: {i}, Real: {y.cpu().numpy()}, Pred: {y_hat.cpu().numpy()}")
    
    # Denormalize predictions and real values
    denormalized_predictions = denormalize_params(np.array(predictions))
    denormalized_real_values = denormalize_params(np.array(real_values))

    # Save denormalized predictions and real values
    pred_df = pd.DataFrame({
        "real": list(denormalized_real_values),
        "pred": list(denormalized_predictions)
    })
    pred_df.to_csv(pred_filename, index=False)

    logging.info(f"Predictions saved to {pred_filename}")

