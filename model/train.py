import torch
import numpy as np
import logging
import os
import pandas as pd
from torch.utils.data import DataLoader
from config.param_config import PARAM_STATS, PARAM_ORDER, denormalize_params
from utils.augmentation import augment_batch

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

def batch_to_device(batch, device):
    return {key: tensor.float().to(device) for key, tensor in batch.items()}

def train(model, train_loader, val_loader, test_loader, loss_fn, opt, args, checkpoint_path):
    # args setting
    num_epochs, test_interval, device = args.num_epochs, args.test_interval, args.device
    accumulation_steps = args.batch_size # gradient accumulation

    # checkpoint setting
    start_epoch = 1
    best_validation_loss = float('inf')
    best_checkpoint_path = os.path.join(os.path.dirname(checkpoint_path), "best_checkpoint.pth")

    # Load checkpoint if exists
    if os.path.isfile(checkpoint_path):
        model, opt, start_epoch, _ = load_checkpoint(model, opt, checkpoint_path, device)
    
    train_losses = []
    val_losses = []

    for epoch_i in range(start_epoch, num_epochs + 1):
        epoch_loss = []
        model.train()
        opt.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch = augment_batch(batch, args.drop_prob) # data augmentation
            batch = batch_to_device(batch, device)
            y = batch['y']
            
            y_hat = model(batch)

            loss = loss_fn(y_hat, y) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                opt.step()
                opt.zero_grad()

            epoch_loss.append(loss.item() * accumulation_steps) # store the full loss
            logging.debug(f"Epoch: {epoch_i}, Train Iteration: {batch_idx + 1}, Loss: {loss.item()*accumulation_steps:.4f}")
        
        avg_train_loss = np.mean(epoch_loss)
        logging.info(f"Epoch: {epoch_i}, Train Loss: {avg_train_loss:.4f}")
        train_losses.append(avg_train_loss)
        
        # Validate the model
        val_loss = validate(model, val_loader, loss_fn, device, epoch_i)
        logging.info(f"Epoch: {epoch_i}, Validation Loss: {val_loss:.4f}")
        val_losses.append(val_loss)
        
        # Save current checkpoint
        save_checkpoint(model, opt, epoch_i, val_loss, checkpoint_path)
        
        # Save the best model if validation loss has improved
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            save_checkpoint(model, opt, epoch_i, best_validation_loss, best_checkpoint_path)
        
        # Save the train and validation losses after each epoch
        loss_dir = os.path.dirname(checkpoint_path)
        pd.DataFrame({"train_loss": train_losses}).to_csv(os.path.join(loss_dir, "train_losses.csv"), index=False)
        pd.DataFrame({"val_loss": val_losses}).to_csv(os.path.join(loss_dir, "val_losses.csv"), index=False)
        
        if not args.tuning and epoch_i % test_interval == 0:
            logging.info(f"Starting evaluation for epoch {epoch_i}")
            # Temporarily load the best checkpoint for evaluation
            current_model_state = model.state_dict()
            current_opt_state = opt.state_dict()
            model, opt, _, _ = load_checkpoint(model, opt, best_checkpoint_path, device)
            evaluate(model, test_loader, device, os.path.join(os.path.dirname(best_checkpoint_path), "pred.txt"), args.target_labels)
            # Restore the current training state
            model.load_state_dict(current_model_state)
            opt.load_state_dict(current_opt_state)
    
    return best_validation_loss

def validate(model, val_loader, loss_fn, device, epoch_i):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = batch_to_device(batch, device)
            y = batch['y']
        
            y_hat = model(batch)
            loss = loss_fn(y_hat, y)
            val_loss.append(loss.item())
            logging.debug(f"Epoch: {epoch_i}, Validation Iteration: {batch_idx + 1}, Loss: {loss.item():.4f}")
    
    return np.mean(val_loss)

def evaluate(model, test_loader, device, pred_filename, target_labels):
    model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch_to_device(batch, device)
            y = batch['y']

            y_hat = model(batch)
            predictions.extend(y_hat.cpu().numpy())
            real_values.append(y.cpu().numpy())
            logging.debug(f"Test Iteration: {batch_idx + 1}, Real: {y.cpu().numpy()}, Pred: {y_hat.cpu().numpy()}")
    
    # Denormalize predictions and real values
    denormalized_predictions = denormalize_params(np.array(predictions), target_labels)
    denormalized_real_values = denormalize_params(np.array(real_values), target_labels)

    # Save denormalized predictions and real values
    pred_df = pd.DataFrame({
        "real": list(denormalized_real_values),
        "pred": list(denormalized_predictions)
    })
    pred_df.to_csv(pred_filename, index=False)

    logging.info(f"Predictions saved to {pred_filename}")
