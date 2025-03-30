import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from scipy.stats import entropy
import seaborn as sns

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def load_npz_to_list(filename):
    """Loads data from an .npz file into a list of numpy arrays."""
    try:
        loaded = np.load(filename, allow_pickle=True)
        # Ensure keys are sorted numerically if they are string numbers
        keys = sorted(loaded.files, key=lambda k: int(k.split('_')[-1]) if k.split('_')[-1].isdigit() else float('inf'))
        return [loaded[key] for key in keys]
    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return []
    except Exception as e:
        print(f"An error occurred while loading {filename}: {e}")
        return []


def calculate_threshold_gcn(model, val_loader, device, criterion, node_features, adj_matrix, percentile=95):
    """Calculates the anomaly threshold based on reconstruction errors on the validation set."""
    model.eval()
    losses = []
    print(f"Calculating threshold using {percentile}th percentile of validation losses...")
    with torch.no_grad():
        # Add tqdm progress bar
        for batch in tqdm(val_loader, desc="Calculating Threshold"):
            seq = batch['sequence'].to(device)
            state_idx = batch['state_index'].to(device)
            # Ensure graph components are on the correct device
            node_features = node_features.to(device)
            adj_matrix = adj_matrix.to(device)

            output = model(seq, state_idx, node_features, adj_matrix)
            # Calculate loss per sequence in the batch
            for i in range(seq.size(0)):
                loss = criterion(output[i].unsqueeze(0), seq[i].unsqueeze(0))
                losses.append(loss.item())

    losses = np.array(losses)
    if len(losses) == 0:
        print("Warning: No losses calculated from validation set. Returning threshold 1.0")
        return 1.0

    # Print loss distribution statistics
    print(f"Validation Loss Stats: Min={np.min(losses):.6f}, Max={np.max(losses):.6f}, Mean={np.mean(losses):.6f}, Median={np.median(losses):.6f}")

    threshold = np.percentile(losses, percentile)
    print(f"Calculated threshold: {threshold:.6f}")
    return threshold

def evaluate_model_gcn(model, test_loader, device, criterion, threshold, node_features, adj_matrix):
    """Evaluates the GCN-Transformer model on the test set."""
    model.eval()
    true_labels = []
    predictions = []
    test_losses = [] # Store test losses
    sample_reconstructions = {'normal': None, 'anomaly': None} # Store one example of each

    print("Evaluating model on test set...")
    with torch.no_grad():
        # Add tqdm progress bar
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            seq = batch['sequence'].to(device)
            labels = batch['label'].cpu().numpy()
            state_idx = batch['state_index'].to(device)
            # Ensure graph components are on the correct device
            node_features = node_features.to(device)
            adj_matrix = adj_matrix.to(device)

            output = model(seq, state_idx, node_features, adj_matrix)
            # Calculate loss per sequence (assuming batch_size=1 for test loader)
            loss = criterion(output, seq)
            current_loss = loss.item()
            test_losses.append(current_loss)

            # Prediction based on threshold
            pred_label = 1 if current_loss > threshold else 0
            predictions.append(pred_label)
            true_labels.extend(labels) # labels should be a list/array

            # Store first anomalous example encountered
            if labels[0] == 1 and sample_reconstructions['anomaly'] is None:
                sample_reconstructions['anomaly'] = {
                    'original': seq[0].cpu().numpy(),
                    'reconstructed': output[0].cpu().numpy(),
                    'loss': current_loss
                }
            # Note: We can't easily get a 'normal' sample here as test_loader contains only anomalies

    print(f"Test Loss Stats: Min={np.min(test_losses):.6f}, Max={np.max(test_losses):.6f}, Mean={np.mean(test_losses):.6f}, Median={np.median(test_losses):.6f}")
    pred_counts = np.unique(predictions, return_counts=True)
    print(f"Prediction Counts: {dict(zip(pred_counts[0], pred_counts[1]))}")


    return np.array(true_labels), np.array(predictions), np.array(test_losses), sample_reconstructions


def plot_reconstructions(val_loader, sample_recons_test, model, device, node_features, adj_matrix, scaler=None, n_examples=1, plots_dir='plots'):
    """Plots original vs. reconstructed signals for normal and anomalous examples."""
    print(f"Generating reconstruction plots to {plots_dir}...")
    plt.style.use('seaborn-v0_8-whitegrid') # Use a modern style

    # Define consistent colors
    original_color = '#1f77b4' # Muted Blue
    reconstructed_color = '#ff7f0e' # Safety Orange
    anomaly_color = '#d62728' # Muted Red for original anomaly

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True) # Slightly adjusted size
    plot_count = 0

    # --- Plot Anomalous Example --- 
    ax1 = axes[0]
    if sample_recons_test['anomaly']:
        original = sample_recons_test['anomaly']['original']
        reconstructed = sample_recons_test['anomaly']['reconstructed']
        loss = sample_recons_test['anomaly']['loss']

        if scaler:
            original = scaler.inverse_transform(original.reshape(-1, 1)).flatten()
            reconstructed = scaler.inverse_transform(reconstructed.reshape(-1, 1)).flatten()

        ax1.plot(original, label='Original Anomalous Signal', color=anomaly_color, linewidth=1.5)
        ax1.plot(reconstructed, label=f'Reconstructed (Loss: {loss:.4f})', color=reconstructed_color, linestyle='--', linewidth=1.5)
        ax1.set_title("Anomalous Dishwasher Cycle vs. Reconstruction", fontsize=14)
        ax1.set_ylabel("Power Consumption", fontsize=12)
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        plot_count += 1
    else:
        print("No anomalous sample found/returned from evaluation to plot.")
        ax1.text(0.5, 0.5, 'No anomalous sample available', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title("Anomalous Dishwasher Cycle vs. Reconstruction", fontsize=14)

    # --- Plot Normal Example --- 
    ax2 = axes[1]
    model.eval()
    normal_found = False
    with torch.no_grad():
        # Only iterate enough to find one normal example
        for batch in val_loader:
            seq = batch['sequence'].to(device)
            state_idx = batch['state_index'].to(device)
            labels = batch['label'].cpu().numpy()
            node_features = node_features.to(device)
            adj_matrix = adj_matrix.to(device)
            output = model(seq, state_idx, node_features, adj_matrix)

            for i in range(seq.size(0)):
                if labels[i] == 0: # Found a normal sample
                    original = seq[i].cpu().numpy()
                    reconstructed = output[i].cpu().numpy()
                    loss = nn.L1Loss()(torch.tensor(reconstructed), torch.tensor(original)).item()

                    if scaler:
                        original = scaler.inverse_transform(original.reshape(-1, 1)).flatten()
                        reconstructed = scaler.inverse_transform(reconstructed.reshape(-1, 1)).flatten()

                    ax2.plot(original, label='Original Normal Signal', color=original_color, linewidth=1.5)
                    ax2.plot(reconstructed, label=f'Reconstructed (Loss: {loss:.4f})', color=reconstructed_color, linestyle='--', linewidth=1.5)
                    ax2.set_title("Normal Dishwasher Cycle vs. Reconstruction", fontsize=14)
                    ax2.set_xlabel("Time Steps", fontsize=12)
                    ax2.set_ylabel("Power Consumption", fontsize=12)
                    ax2.legend()
                    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
                    normal_found = True
                    plot_count += 1
                    break
            if normal_found:
                break

    if not normal_found:
        print("Could not find a normal sample in the validation set to plot.")
        ax2.text(0.5, 0.5, 'No normal sample found in validation set', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title("Normal Dishwasher Cycle vs. Reconstruction", fontsize=14)

    # --- Final Touches --- 
    fig.tight_layout(pad=2.0)
    
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "reconstruction_examples_gcn.png")
    plt.savefig(save_path)
    print(f"Reconstruction plots saved to {save_path}")
    plt.close(fig) # Close the specific figure

# --- Original Transformer Utilities (Keep if needed, otherwise remove) ---
# def calculate_threshold(...)
# def evaluate_model(...)
# def plot_loss(...) # Maybe keep this one if useful? Or integrate into train loop plot
# def plot_reconstruction_error(...) # May replace with new plot

# Example: Keep plot_loss if you still want a separate loss plot function
def plot_loss(train_losses, val_losses, filename='loss_plot_transformer.png'):
    # ... (keep existing code for this if desired) ...
    pass # Placeholder if removing 