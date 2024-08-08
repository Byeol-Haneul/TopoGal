import torch
import numpy as np
import logging
import os
import pandas as pd
from torch.utils.data import DataLoader
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

def train(model, train_loader, val_loader, test_loader, loss_fn, opt, args, checkpoint_path):
    # args setting
    num_epochs, test_interval, device = args.num_epochs, args.test_interval, args.device,

    # checkpoint setting
    start_epoch = 1
    best_loss = float('inf')
    best_checkpoint_path = os.path.join(os.path.dirname(checkpoint_path), "best_checkpoint.pth")

    # Load checkpoint if exists
    if os.path.isfile(checkpoint_path):
        model, opt, start_epoch, _ = load_checkpoint(model, opt, checkpoint_path, device)
    
    train_losses = []
    val_losses = []

    for epoch_i in range(start_epoch, num_epochs + 1):
        epoch_loss = []
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            x_0 = batch['x_0'].float().to(device)
            x_1 = batch['x_1'].float().to(device)
            x_2 = batch['x_2'].float().to(device)
            x_3 = batch['x_3'].float().to(device)
            n0_to_0 = batch['n0_to_0'].float().to(device)
            n1_to_1 = batch['n1_to_1'].float().to(device)
            n2_to_2 = batch['n2_to_2'].float().to(device)
            n3_to_3 = batch['n3_to_3'].float().to(device)
            n0_to_1 = batch['n0_to_1'].float().to(device)
            n1_to_2 = batch['n1_to_2'].float().to(device)
            n2_to_3 = batch['n2_to_3'].float().to(device)
            global_feature = batch['global_feature'].float().to(device)
            y = batch['y'].float().to(device)
            
            opt.zero_grad()
            y_hat = model(x_0, x_1, x_2, x_3, n0_to_0, n1_to_1, n2_to_2, n3_to_3, n0_to_1, n1_to_2, n2_to_3, global_feature)

            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())
            logging.debug(f"Epoch: {epoch_i}, Train Iteration: {batch_idx + 1}, Loss: {loss.item():.4f}")
        
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
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, opt, epoch_i, best_loss, best_checkpoint_path)
        
        # Save the train and validation losses after each epoch
        loss_dir = os.path.dirname(checkpoint_path)
        pd.DataFrame({"train_loss": train_losses}).to_csv(os.path.join(loss_dir, "train_losses.csv"), index=False)
        pd.DataFrame({"val_loss": val_losses}).to_csv(os.path.join(loss_dir, "val_losses.csv"), index=False)
        
        if epoch_i % test_interval == 0:
            logging.info(f"Starting evaluation for epoch {epoch_i}")
            # Temporarily load the best checkpoint for evaluation
            current_model_state = model.state_dict()
            current_opt_state = opt.state_dict()
            model, opt, _, _ = load_checkpoint(model, opt, best_checkpoint_path, device)
            evaluate(model, test_loader, device, os.path.join(os.path.dirname(best_checkpoint_path), "pred.txt"))
            # Restore the current training state
            model.load_state_dict(current_model_state)
            opt.load_state_dict(current_opt_state)

def validate(model, val_loader, loss_fn, device, epoch_i):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x_0 = batch['x_0'].float().to(device)
            x_1 = batch['x_1'].float().to(device)
            x_2 = batch['x_2'].float().to(device)
            x_3 = batch['x_3'].float().to(device)
            n0_to_0 = batch['n0_to_0'].float().to(device)
            n1_to_1 = batch['n1_to_1'].float().to(device)
            n2_to_2 = batch['n2_to_2'].float().to(device)
            n3_to_3 = batch['n3_to_3'].float().to(device)
            n0_to_1 = batch['n0_to_1'].float().to(device)
            n1_to_2 = batch['n1_to_2'].float().to(device)
            n2_to_3 = batch['n2_to_3'].float().to(device)
            global_feature = batch['global_feature'].float().to(device)
            y = batch['y'].float().to(device)
            
            y_hat = model(x_0, x_1, x_2, x_3, n0_to_0, n1_to_1, n2_to_2, n3_to_3, n0_to_1, n1_to_2, n2_to_3, global_feature)
            loss = loss_fn(y_hat, y)
            val_loss.append(loss.item())
            logging.debug(f"Epoch: {epoch_i}, Validation Iteration: {batch_idx + 1}, Loss: {loss.item():.4f}")
    
    return np.mean(val_loss)

def evaluate(model, test_loader, device, pred_filename):
    model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x_0 = batch['x_0'].float().to(device)
            x_1 = batch['x_1'].float().to(device)
            x_2 = batch['x_2'].float().to(device)
            x_3 = batch['x_3'].float().to(device)
            n0_to_0 = batch['n0_to_0'].float().to(device)
            n1_to_1 = batch['n1_to_1'].float().to(device)
            n2_to_2 = batch['n2_to_2'].float().to(device)
            n3_to_3 = batch['n3_to_3'].float().to(device)
            n0_to_1 = batch['n0_to_1'].float().to(device)
            n1_to_2 = batch['n1_to_2'].float().to(device)
            n2_to_3 = batch['n2_to_3'].float().to(device)
            global_feature = batch['global_feature'].float().to(device)
            y = batch['y'].float().to(device)

            y_hat = model(x_0, x_1, x_2, x_3, n0_to_0, n1_to_1, n2_to_2, n3_to_3, n0_to_1, n1_to_2, n2_to_3, global_feature)
            predictions.extend(y_hat.cpu().numpy())
            real_values.append(y.cpu().numpy())
            logging.debug(f"Test Iteration: {batch_idx + 1}, Real: {y.cpu().numpy()}, Pred: {y_hat.cpu().numpy()}")
    
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
