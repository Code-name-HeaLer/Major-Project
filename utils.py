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
    """Calculates the anomaly threshold based on reconstruction errors on the validation set.
       Uses the *mean* reconstruction error across the sequence for thresholding.
    """
    model.eval()
    losses = []
    print(f"Calculating threshold using {percentile}th percentile of validation losses...")

    # Ensure criterion has reduction='none' to get per-element losses first
    # We'll create a local criterion if the passed one isn't correct
    if criterion.reduction != 'none':
        print(f"Warning: Criterion passed to calculate_threshold_gcn has reduction='{criterion.reduction}'. Temporarily using reduction='none' internally.")
        loss_fn_none = nn.L1Loss(reduction='none') # Assuming L1Loss, adjust if different
    else:
        loss_fn_none = criterion

    with torch.no_grad():
        # Add tqdm progress bar
        for batch in tqdm(val_loader, desc="Calculating Threshold"):
            seq = batch['sequence'].to(device)
            state_idx = batch['state_index'].to(device)
            # Ensure graph components are on the correct device
            node_features = node_features.to(device)
            adj_matrix = adj_matrix.to(device)

            output = model(seq, state_idx, node_features, adj_matrix)

            # Calculate per-element loss for the batch
            loss_per_element = loss_fn_none(output, seq) # Shape: [batch, seq_len, features]

            # Calculate the mean loss *per sequence* in the batch
            # Mean across sequence length and feature dimensions
            mean_loss_per_sequence = torch.mean(loss_per_element, dim=(1, 2)) # Shape: [batch]

            # Append scalar losses to the list
            losses.extend(mean_loss_per_sequence.cpu().numpy())


    losses = np.array(losses)
    if len(losses) == 0:
        print("Warning: No losses calculated from validation set. Returning threshold 1.0")
        return 1.0

    # Print loss distribution statistics
    print(f"Validation Loss Stats (Mean Per Sequence): Min={np.min(losses):.6f}, Max={np.max(losses):.6f}, Mean={np.mean(losses):.6f}, Median={np.median(losses):.6f}")

    threshold = np.percentile(losses, percentile)
    print(f"Calculated threshold: {threshold:.6f}")
    return threshold

def evaluate_model_gcn(model, test_loader, device, criterion, threshold, node_features, adj_matrix):
    """Evaluates the GCN-Transformer model on the test set.

    Args:
        model: The trained GCN-Transformer model.
        test_loader: DataLoader for the test set (batch size should be 1).
        device: The device (CPU or GPU) to run evaluation on.
        criterion: Loss function (e.g., nn.L1Loss with reduction='mean' for average loss reporting).
        threshold: Anomaly detection threshold.
        node_features: Tensor of node features.
        adj_matrix: Tensor of the adjacency matrix.

    Returns:
        true_labels: Numpy array of true labels (0 for normal, 1 for anomaly).
        predictions: Numpy array of predicted labels (0 or 1).
        test_losses: Numpy array of reconstruction losses for each test sample.
        sample_reconstructions: Deprecated, replaced by all_results.
        all_results: A list of dictionaries, each containing detailed info for one test sample:
                     {'original', 'reconstructed', 'loss', 'true_label', 
                      'pred_label', 'state_index', 'anomaly_type'}
    """
    model.eval()
    true_labels = []
    predictions = []
    test_losses = []
    all_results = [] # New list to store detailed results for each sample
    # sample_reconstructions is removed

    # Ensure graph components are on the correct device once before the loop
    node_features = node_features.to(device)
    adj_matrix = adj_matrix.to(device)

    print("Evaluating model on test set...")
    # Use a loss function with reduction='none' internally to get per-sample loss easily
    # even if the passed criterion has reduction='mean' for average reporting.
    loss_fn_none = nn.L1Loss(reduction='none')

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            seq = batch['sequence'].to(device) # Shape: [1, seq_len, input_dim]
            label = batch['label'].cpu().numpy()[0] # Get single label
            state_idx = batch['state_index'].to(device) # Shape: [1]
            anomaly_type = batch['anomaly_type'][0] # Get single anomaly type string

            output = model(seq, state_idx, node_features, adj_matrix)

            # Calculate loss for this specific sequence
            # Use loss_fn_none and then calculate the mean for the single item batch
            loss_per_element = loss_fn_none(output, seq)
            current_loss = torch.mean(loss_per_element).item()
            test_losses.append(current_loss)

            # Prediction based on threshold
            pred_label = 1 if current_loss > threshold else 0
            predictions.append(pred_label)
            true_labels.append(label)

            # Store detailed results for this sample
            all_results.append({
                'original': seq[0].cpu().numpy(), # Get the sequence data [seq_len, input_dim]
                'reconstructed': output[0].cpu().numpy(),
                'loss': current_loss,
                'true_label': label,
                'pred_label': pred_label,
                'state_index': state_idx[0].item(), # Get scalar state index
                'anomaly_type': anomaly_type
            })

    test_losses = np.array(test_losses)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Optional: Print stats only if there are results
    if len(test_losses) > 0:
        print(f"Test Loss Stats: Min={np.min(test_losses):.6f}, Max={np.max(test_losses):.6f}, Mean={np.mean(test_losses):.6f}, Median={np.median(test_losses):.6f}")
    if len(predictions) > 0:
        pred_counts = np.unique(predictions, return_counts=True)
        print(f"Prediction Counts: {dict(zip(pred_counts[0], pred_counts[1]))}")
    else:
        print("No predictions were made (test set might be empty).")

    # Return the comprehensive results list instead of sample_reconstructions
    return true_labels, predictions, test_losses, None, all_results # Return None for deprecated arg


def plot_reconstructions(all_results, scaler=None, plots_dir='plots'):
    """Plots original vs. reconstructed signals for normal and each type of anomalous example.

    Args:
        all_results: List of dictionaries from evaluate_model_gcn.
        scaler: Fitted MinMaxScaler or None, for inverse transforming data.
        plots_dir: Directory to save the plot.
    """
    if not all_results:
        print("No results available to generate reconstruction plots.")
        return

    print(f"Generating reconstruction plots to {plots_dir}...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Define consistent colors
    colors = {
        'normal_orig': '#1f77b4', # Muted Blue
        'normal_recon': '#aec7e8', # Light Blue
        'anomaly_orig': '#d62728', # Muted Red
        'anomaly_recon': '#ff9896', # Light Red
        'recon_default': '#ff7f0e' # Safety Orange (for general reconstruction line)
    }

    # Find one example for each category (normal, high_energy, repeated_cycle, noisy, etc.)
    examples_to_plot = {}
    anomaly_types_found = set()

    # Prioritize finding one of each specific anomaly type first
    for result in all_results:
        a_type = result['anomaly_type']
        if a_type != 'normal' and a_type not in examples_to_plot:
            examples_to_plot[a_type] = result
            anomaly_types_found.add(a_type)

    # Then find one normal example
    if 'normal' not in examples_to_plot:
        for result in all_results:
            if result['anomaly_type'] == 'normal':
                examples_to_plot['normal'] = result
                break # Found one normal, stop looking

    # Determine the number of plots needed (1 normal + number of unique anomaly types found)
    plot_categories = ['normal'] + sorted(list(anomaly_types_found))
    n_plots = len(plot_categories)

    if n_plots == 0:
        print("No normal or anomalous examples found in results to plot.")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
    # Ensure axes is always an array, even if n_plots is 1
    if n_plots == 1:
        axes = [axes]

    plot_count = 0
    for i, category in enumerate(plot_categories):
        ax = axes[i]
        if category in examples_to_plot:
            example = examples_to_plot[category]
            original = example['original']
            reconstructed = example['reconstructed']
            loss = example['loss']
            true_label = example['true_label'] # 0 for normal, 1 for anomaly

            # Inverse transform if scaler is provided
            if scaler:
                original_shape = original.shape
                reconstructed_shape = reconstructed.shape
                try:
                    original = scaler.inverse_transform(original.reshape(-1, 1)).reshape(original_shape).flatten()
                    reconstructed = scaler.inverse_transform(reconstructed.reshape(-1, 1)).reshape(reconstructed_shape).flatten()
                except Exception as e:
                    print(f"Warning: Error inverse transforming data for plotting category '{category}': {e}")
                    # Continue with potentially scaled data

            # Determine plot title and colors based on category
            if category == 'normal':
                title = "Normal Dishwasher Cycle vs. Reconstruction"
                orig_color = colors['normal_orig']
                recon_style = '--'
            else:
                # Capitalize anomaly type for title
                title_cat = category.replace('_', ' ').title()
                title = f"Anomalous ({title_cat}) Cycle vs. Reconstruction"
                orig_color = colors['anomaly_orig']
                recon_style = '--' # Could use different style if needed

            ax.plot(original, label=f'Original {category.title()} Signal', color=orig_color, linewidth=1.5)
            ax.plot(reconstructed, label=f'Reconstructed (Loss: {loss:.4f})', color=colors['recon_default'], linestyle=recon_style, linewidth=1.5)
            ax.set_title(title, fontsize=14)
            ax.set_ylabel("Power Consumption", fontsize=12)
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            plot_count += 1
        else:
            # Should not happen if logic above is correct, but handle defensively
            ax.text(0.5, 0.5, f'No example found for \'{category}\'', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f"{category.title()} Example", fontsize=14)

    # Set common xlabel only on the last plot
    if n_plots > 0:
        axes[-1].set_xlabel("Time Steps", fontsize=12)

    fig.tight_layout(pad=2.0)

    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "reconstruction_examples_gcn.png")
    plt.savefig(save_path)
    print(f"Reconstruction plots saved to {save_path}")
    plt.close(fig)

# --- Placeholder for plot_cluster_centroids ---
def plot_cluster_centroids(kmeans_model, node_features, scaler=None, plots_dir='plots', n_clusters_to_show=None):
    """Plots the cluster centroids (mean windows)."""
    print(f"Generating cluster centroid plots to {plots_dir}...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Node features are expected as [n_clusters, seq_len, input_dim]
    if node_features is None or kmeans_model is None:
        print("Node features or kmeans model not available. Cannot plot centroids.")
        return

    centroids = node_features.cpu().numpy() # Get centroids from node features
    n_clusters = centroids.shape[0]
    seq_len = centroids.shape[1]
    input_dim = centroids.shape[2]

    if input_dim != 1:
        print(f"Warning: Plotting centroids assumes input_dim=1, but got {input_dim}. Plot may be incorrect.")

    if n_clusters_to_show is None or n_clusters_to_show > n_clusters:
        n_clusters_to_show = n_clusters

    # Determine grid size (e.g., 3 columns wide)
    n_cols = 3
    n_rows = (n_clusters_to_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True, squeeze=False)
    fig.suptitle("Power Signature Patterns for Each Cluster Node", fontsize=16, y=1.02)

    for i in range(n_clusters_to_show):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        centroid_data = centroids[i, :, 0] # Get the first feature dimension

        # Inverse transform if scaler is provided
        if scaler:
             try:
                 centroid_data = scaler.inverse_transform(centroid_data.reshape(-1, 1)).flatten()
             except Exception as e:
                 print(f"Warning: Error inverse transforming centroid {i}: {e}")

        ax.plot(centroid_data, linewidth=1.5)
        ax.set_title(f"Cluster Node {i}")
        ax.grid(True, linestyle='--', linewidth=0.5)
        if col == 0:
            ax.set_ylabel("Power Value")
        if row == n_rows - 1:
            ax.set_xlabel("Time Step Index")

    # Hide unused subplots
    for i in range(n_clusters_to_show, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        fig.delaxes(axes[row, col])

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "cluster_centroids.png")
    plt.savefig(save_path)
    print(f"Cluster centroid plot saved to {save_path}")
    plt.close(fig)

# --- Placeholder for plot_sample_windows_vs_centroids ---
def plot_sample_windows_vs_centroids(all_results, node_features, scaler=None, plots_dir='plots', n_samples=10):
    """Plots sample windows against their assigned cluster centroids."""
    print(f"Generating sample window vs. centroid plots to {plots_dir}...")
    plt.style.use('seaborn-v0_8-whitegrid')

    if not all_results or node_features is None:
        print("Results or node features not available. Cannot plot window vs centroids.")
        return

    centroids = node_features.cpu().numpy()
    n_clusters = centroids.shape[0]
    input_dim = centroids.shape[2]

    if input_dim != 1:
        print(f"Warning: Plotting assumes input_dim=1, but got {input_dim}. Plot may be incorrect.")

    # Select n_samples randomly (or first n_samples) from all_results
    # Ensure we don't request more samples than available
    actual_n_samples = min(n_samples, len(all_results))
    if actual_n_samples < n_samples:
        print(f"Warning: Requested {n_samples} samples, but only {actual_n_samples} available in results.")

    # Randomly sample indices if results are many, otherwise take the first few
    if len(all_results) > actual_n_samples:
        indices_to_plot = np.random.choice(len(all_results), actual_n_samples, replace=False)
    else:
        indices_to_plot = np.arange(actual_n_samples)

    samples_to_plot = [all_results[i] for i in indices_to_plot]

    # Determine grid size (e.g., 5 columns wide)
    n_cols = 5
    n_rows = (actual_n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True, squeeze=False)
    fig.suptitle("Sample Windows vs. Assigned Cluster Centroids", fontsize=16, y=1.02)

    for i, sample_info in enumerate(samples_to_plot):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        window_data = sample_info['original'][:, 0] # Assume input_dim=1
        state_index = sample_info['state_index']
        anomaly_type = sample_info['anomaly_type']
        true_label = sample_info['true_label']

        # Get the corresponding centroid
        if 0 <= state_index < n_clusters:
            centroid_data = centroids[state_index, :, 0]
        else:
            print(f"Warning: Invalid state index {state_index} for sample {i}. Skipping centroid plot.")
            centroid_data = np.zeros_like(window_data) # Plot zeros as placeholder

        # Inverse transform if scaler is provided
        if scaler:
            try:
                window_data = scaler.inverse_transform(window_data.reshape(-1, 1)).flatten()
                centroid_data = scaler.inverse_transform(centroid_data.reshape(-1, 1)).flatten()
            except Exception as e:
                print(f"Warning: Error inverse transforming data for sample vs centroid plot {i}: {e}")

        # Plot
        ax.plot(window_data, label=f'Actual Window Signal ({anomaly_type.title()})', color='#1f77b4', linewidth=1.5)
        ax.plot(centroid_data, label=f'Centroid Node {state_index}', color='#d62728', linestyle='--', linewidth=1.5)

        # Use original sample index if available, otherwise use plot index `i`
        # This requires `all_results` to potentially include an original index if needed.
        # For now, just use the plot index i.
        # sample_orig_idx = sample_info.get('original_index', i)
        title = f'Window {i} ({anomaly_type.title()})\nAssigned to Node {state_index}'
        ax.set_title(title)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.legend()

        if col == 0:
            ax.set_ylabel("Power Value")
        if row == n_rows - 1:
            ax.set_xlabel("Time Step Index")

    # Hide unused subplots
    for i in range(actual_n_samples, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        fig.delaxes(axes[row, col])

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "sample_window_vs_centroids.png")
    plt.savefig(save_path)
    print(f"Sample window vs centroid plot saved to {save_path}")
    plt.close(fig)


# --- Original Transformer Utilities (Keep if needed, otherwise remove) ---
# def calculate_threshold(...)
# def evaluate_model(...)
# def plot_loss(...) # Maybe keep this one if useful? Or integrate into train loop plot
# def plot_reconstruction_error(...) # May replace with new plot

# Example: Keep plot_loss if you still want a separate loss plot function
def plot_loss(train_losses, val_losses, filename='loss_plot_transformer.png'):
    # ... (keep existing code for this if desired) ...
    pass # Placeholder if removing 