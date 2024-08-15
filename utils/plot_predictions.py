import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.metrics import r2_score
import argparse

dpi = 200
matplotlib.rcParams['figure.dpi'] = dpi
matplotlib.rcParams.update({'font.size': 8})

def parse_predictions(filename):
    df = pd.read_csv(filename, delimiter=',', header=0, index_col=False)
    real_str = df['real'].values
    pred_str = df['pred'].values
    real = [np.fromstring(item.strip('[]'), sep=' ') for item in real_str]
    pred = [np.fromstring(item.strip('[]'), sep=' ') for item in pred_str]
    return real, pred

def calculate_chi_square(y, y_hat, yerr):
    chi_square = np.mean(((y - y_hat) / yerr) ** 2)
    return chi_square

def calculate_r2(y, y_hat):
    return r2_score(y, y_hat)

def calculate_relative_error(y, y_hat):
    relative_error = np.mean(np.abs((y - y_hat) / y) * 100)
    return relative_error

def plot_pred(directory, title=None, save=False):
    filename = os.path.join(directory, 'pred.txt')
    try:
        y_values, y_hat_values = parse_predictions(filename)
    except FileNotFoundError:
        print("NO PREDICTION MADE YET")
        return

    y_values = np.array(y_values)
    y_hat_values = np.array(y_hat_values)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    if title is not None:
        fig.suptitle(title)
    
    titles = [r'$\Omega_{\rm m}$', r'$\sigma_{8}$', r'$A_{\rm SN1}$', r'$A_{\rm AGN1}$', r'$A_{\rm SN2}$', r'$A_{\rm AGN2}$']

    for i in range(6):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        
        if y_hat_values.shape[1] == 12:
            yerr = abs(y_hat_values[:, i + 6])
            y_hat = y_hat_values[:, i]
            ax.errorbar(x=y_values[:, i], y=y_hat, yerr=yerr, fmt='o', color="blue", ecolor="blue", markersize=4, elinewidth=0.5)
            chi_square = calculate_chi_square(y_values[:, i], y_hat, yerr)
        else:
            y_hat = y_hat_values[:, i]
            ax.scatter(y_values[:, i], y_hat, c='blue', marker='o', s=4, label='Prediction')
            chi_square = None
        
        r2 = calculate_r2(y_values[:, i], y_hat)
        relative_error = calculate_relative_error(y_values[:, i], y_hat)
        
        ax.plot([-1, 10], [-1, 10], 'r--', label='Ideal Fit')
        
        ax.set_xlim(min(y_values[:, i]), max(y_values[:, i]))
        ax.set_ylim(min(y_values[:, i]), max(y_values[:, i]))
    
        ax.set_xlabel(r'Actual values ($y$)')
        ax.set_ylabel(r'Predicted values ($\hat{y}$)')
        ax.set_title(titles[i])
        ax.legend()
        ax.grid(True)
        
        metrics_text = f"$R^2$ = {r2:.2f}\n" + (r"$\chi^{2}$"+f" = {chi_square:.2f}\n" if chi_square is not None else "") + r"$\epsilon$"+f" = {relative_error:.2f}%"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    if save:
        save_path = os.path.join(directory, 'pred_plot.png')
        plt.savefig(save_path)
        print(f"Prediction plot saved to {save_path}")
    plt.show()

def plot_Om(directory, target_labels=None, title=None, save=False):
    filename = os.path.join(directory, 'pred.txt')
    try:
        y_values, y_hat_values = parse_predictions(filename)
    except FileNotFoundError:
        print("NO PREDICTION MADE YET")
        return

    y_values = np.array(y_values)
    y_hat_values = np.array(y_hat_values)

    fig, ax = plt.subplots(figsize=(3, 3))
    if title is not None:
        fig.suptitle(title)
    
    i = 0
    num_params = 1 if target_labels is None else len(target_labels)
    
    if y_hat_values.shape[1] == 12:
        yerr = abs(y_hat_values[:, i + 6])
        y_hat = y_hat_values[:, i]
    else:
        yerr = abs(y_hat_values[:, i + num_params])
        y_hat = y_hat_values[:, i]

    ax.errorbar(x=y_values[:, i], y=y_hat, yerr=yerr, fmt='o', color="blue", ecolor="blue", markersize=4, elinewidth=0.5)
    chi_square = calculate_chi_square(y_values[:, i], y_hat, yerr)
    
    r2 = calculate_r2(y_values[:, i], y_hat)
    relative_error = calculate_relative_error(y_values[:, i], y_hat)
    
    ax.plot([-1, 10], [-1, 10], 'r--', label='Ideal Fit')
    
    ax.set_xlim(min(y_values[:, i]), max(y_values[:, i]))
    ax.set_ylim(min(y_values[:, i]), max(y_values[:, i]))
    
    ax.set_xlabel(r'Actual values ($y$)')
    ax.set_ylabel(r'Predicted values ($\hat{y}$)')
    ax.set_title(r'$\Omega_{\rm m}$')
    ax.legend()
    ax.grid(True)
    
    metrics_text = f"$R^2$ = {r2:.2f}\n" + (r"$\chi^{2}$"+f" = {chi_square:.2f}\n" if chi_square is not None else "") + r"$\epsilon$"+f" = {relative_error:.2f}%"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    if save:
        save_path = os.path.join(directory, 'Om_plot.png')
        plt.savefig(save_path)
        print(f"Om plot saved to {save_path}")
    plt.show()

def plot_loss(directory, save=False):
    fig, ax = plt.subplots(figsize=(3, 3))
    train = np.loadtxt(os.path.join(directory, "train_losses.csv"), skiprows=1)
    validation = np.loadtxt(os.path.join(directory, "val_losses.csv"), skiprows=1)
    ax.plot(train, label="Training Loss")
    ax.plot(validation, label="Validation Loss")
    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    
    plt.tight_layout()
    if save:
        save_path = os.path.join(directory, 'loss_plot.png')
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot prediction results and losses.")
    parser.add_argument('checkpoint_dir', type=str, help="Directory containing prediction results and losses.")
    parser.add_argument('--title', type=str, default=None, help="Title for the plots.")
    parser.add_argument('--save', action='store_true', help="Flag to save the plots instead of only displaying them.")
    args = parser.parse_args()

    plot_pred(args.checkpoint_dir, title=args.title, save=args.save)
    plot_Om(args.checkpoint_dir, title=args.title, save=args.save)
    plot_loss(args.checkpoint_dir, save=args.save)
