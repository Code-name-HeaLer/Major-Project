import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from utils import load_npz_to_list
# Import GCNConv directly from model to check availability
from model import GCNConv # Changed from GCNTransformerAutoencoder
if GCNConv is not None: # Check the imported variable directly
    from graph_utils import construct_graph_components, calculate_entropy, get_window_state
else:
    # Define dummy functions if torch_geometric is not available
    print("Defining dummy graph functions as PyTorch Geometric is not installed.")
    def calculate_entropy(*args, **kwargs):
        return 0.0
    def construct_graph_components(*args, **kwargs):
        return None, None, None
    def get_window_state(*args, **kwargs):
        return 0

class DishwasherDataset(Dataset):
    """PyTorch Dataset for Dishwasher Activations."""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        # Ensure sequence is float32
        return {'sequence': torch.tensor(sequence, dtype=torch.float32), 'label': torch.tensor(label, dtype=torch.long)}

class DishwasherDatasetGCN(Dataset):
    """PyTorch Dataset including sequence state index."""
    def __init__(self, sequences, labels, state_indices):
        self.sequences = sequences
        self.labels = labels
        self.state_indices = state_indices

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        state_index = self.state_indices[idx]
        # Ensure sequence is float32 and state_index is long
        return {
            'sequence': torch.tensor(sequence, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'state_index': torch.tensor(state_index, dtype=torch.long)
        }

def _pad_or_truncate(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Pads or truncates a sequence to the target length."""
    current_len = seq.shape[0]
    if current_len == target_len:
        return seq
    elif current_len > target_len:
        return seq[:target_len, :]
    else:
        padding_size = target_len - current_len
        padding = np.zeros((padding_size, seq.shape[1]), dtype=seq.dtype)
        return np.vstack((seq, padding))

def get_dataloaders(data_dir: str, appliance_col: int = 2, seq_len: int = None,
                  batch_size: int = 32, val_split: float = 0.2, test_data_type: str = 'high_energy',
                  scale_data: bool = True):
    """
    Loads healthy and unhealthy data, preprocesses, splits, and creates DataLoaders.

    Args:
        data_dir: Directory containing the .npz files.
        appliance_col: Index of the column containing appliance energy data (default: 2).
        seq_len: Target sequence length. If None, determines max length from healthy data.
        batch_size: Batch size for DataLoaders.
        val_split: Fraction of healthy data to use for validation.
        test_data_type: Type of unhealthy data to load ('high_energy', 'short_cycle', 'delayed_start').
        scale_data: Whether to apply MinMaxScaler to the appliance energy data.

    Returns:
        train_loader, val_loader, test_loader, scaler (if scale_data is True else None), target_seq_len
    """
    # --- Load Healthy Data ---
    healthy_file = os.path.join(data_dir, "11_REFIT_B2_DW_healthy_activations.npz")
    healthy_activations_list = load_npz_to_list(healthy_file)
    if not healthy_activations_list:
        raise ValueError(f"Could not load healthy data from {healthy_file}")

    # Extract the appliance energy column
    all_healthy_sequences_raw = [act[:, appliance_col:appliance_col+1] for act in healthy_activations_list]

    # Determine target sequence length
    if not seq_len:
        # Use max length from healthy data if not specified
        target_seq_len = max(s.shape[0] for s in all_healthy_sequences_raw)
        print(f"Determined target sequence length from data (max): {target_seq_len}")
    else:
        target_seq_len = seq_len
        print(f"Using specified target sequence length: {target_seq_len}")

    # Pad/Truncate healthy sequences
    healthy_sequences_processed = [
        _pad_or_truncate(seq, target_seq_len) for seq in all_healthy_sequences_raw
    ]

    # Convert to NumPy array *after* ensuring consistent length
    healthy_sequences = np.array(healthy_sequences_processed) # Shape: [num_samples, target_seq_len, 1]
    healthy_labels = np.zeros(len(healthy_sequences)) # Label healthy data as 0

    # --- Scale Data (Optional) ---
    scaler = None
    if scale_data:
        original_shape = healthy_sequences.shape
        # Scaler expects 2D array [samples*seq_len, features]
        data_to_scale = healthy_sequences.reshape(-1, original_shape[-1])
        if data_to_scale.size > 0:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data_to_scale)
            healthy_sequences = scaled_data.reshape(original_shape)
            print("Applied MinMaxScaler to healthy data.")
        else:
            print("Warning: Healthy data is empty after processing, skipping scaling.")

    # --- Split Healthy Data (Train/Validation) ---
    X_train, X_val, y_train, y_val = train_test_split(
        healthy_sequences, healthy_labels, test_size=val_split, random_state=42, shuffle=True
    )

    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # --- Load Unhealthy (Test) Data ---
    if test_data_type == 'high_energy':
        unhealthy_file = os.path.join(data_dir, "11_REFIT_B2_DW_unhealthy_high_energy_activations.npz")
    # Add elif blocks for other types if needed
    else:
        raise ValueError(f"Unsupported test_data_type: {test_data_type}")

    unhealthy_activations_list = load_npz_to_list(unhealthy_file)
    if not unhealthy_activations_list:
        print(f"Warning: Could not load unhealthy data from {unhealthy_file}. Test loader will be empty.")
        # Create an empty array with the correct shape if test data is missing
        X_test = np.empty((0, target_seq_len, 1), dtype=np.float32)
        y_test = np.array([], dtype=np.int64)
    else:
        X_test_raw = [act[:, appliance_col:appliance_col+1] for act in unhealthy_activations_list]
        # Pad/Truncate test sequences
        X_test_processed = [
             _pad_or_truncate(seq, target_seq_len) for seq in X_test_raw
        ]
        X_test = np.array(X_test_processed)
        y_test = np.ones(len(X_test)) # Label unhealthy data as 1

        # --- Scale Test Data (using scaler fitted on train) ---
        if scale_data and scaler and X_test.size > 0:
            original_shape_test = X_test.shape
            data_to_scale_test = X_test.reshape(-1, original_shape_test[-1])
            scaled_data_test = scaler.transform(data_to_scale_test) # Use transform, not fit_transform
            X_test = scaled_data_test.reshape(original_shape_test)
            print("Applied MinMaxScaler to test data.")
        elif scale_data and not scaler:
             print("Warning: scale_data is True, but scaler was not fitted (maybe no training data?). Test data not scaled.")

    print(f"Test set size: {len(X_test)}")

    # --- Create Datasets and DataLoaders ---
    train_dataset = DishwasherDataset(X_train, y_train)
    val_dataset = DishwasherDataset(X_val, y_val)
    test_dataset = DishwasherDataset(X_test, y_test)

    num_workers = 0 # Start with 0 for stability, especially on Windows

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # Use batch_size=1 for test loader to evaluate sequences individually
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    # Return the scaler and the actual sequence length used
    return train_loader, val_loader, test_loader, scaler, target_seq_len

def get_dataloaders_gcn(data_dir: str, appliance_col: int = 2, seq_len: int = None,
                        batch_size: int = 32, val_split: float = 0.2, test_data_type: str = 'high_energy',
                        scale_data: bool = True, n_clusters: int = 10):
    """
    Loads data, performs graph construction on training data, preprocesses, splits,
    calculates state indices, and creates DataLoaders for the GCN-Transformer model.

    Args:
        data_dir, appliance_col, seq_len, batch_size, val_split, test_data_type, scale_data: (as before)
        n_clusters: Number of states/nodes for graph construction.

    Returns:
        train_loader, val_loader, test_loader,
        scaler, kmeans_model, node_features, adj_matrix,
        target_seq_len, input_dim
    """
    if GCNConv is None: # Check the imported variable directly
        raise ImportError("PyTorch Geometric required for GCN data loading.")

    # --- Load Healthy Data ---
    healthy_file = os.path.join(data_dir, "11_REFIT_B2_DW_healthy_activations.npz")
    healthy_activations_list = load_npz_to_list(healthy_file)
    if not healthy_activations_list:
        raise ValueError(f"Could not load healthy data from {healthy_file}")

    all_healthy_sequences_raw = [act[:, appliance_col:appliance_col+1] for act in healthy_activations_list]
    if not all_healthy_sequences_raw: # Check if list is empty
         raise ValueError("No healthy sequences found after extracting appliance column.")
    input_dim = all_healthy_sequences_raw[0].shape[1]

    # Determine target sequence length
    if not seq_len:
        target_seq_len = max(s.shape[0] for s in all_healthy_sequences_raw)
        print(f"Determined target sequence length from data (max): {target_seq_len}")
    else:
        target_seq_len = seq_len
        print(f"Using specified target sequence length: {target_seq_len}")

    # Pad/Truncate healthy sequences
    healthy_sequences_processed = [
        _pad_or_truncate(seq, target_seq_len) for seq in all_healthy_sequences_raw
    ]
    healthy_sequences_full = np.array(healthy_sequences_processed)
    healthy_labels_full = np.zeros(len(healthy_sequences_full))

    # --- Scale Data (Optional, before graph construction based on entropy) ---
    # Scaling might affect entropy, decide if scaling happens before or after entropy calc.
    # Let's scale first, as features (power) are used for node means.
    scaler = None
    if scale_data:
        original_shape = healthy_sequences_full.shape
        data_to_scale = healthy_sequences_full.reshape(-1, original_shape[-1])
        if data_to_scale.size > 0:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data_to_scale)
            healthy_sequences_full = scaled_data.reshape(original_shape)
            print("Applied MinMaxScaler to healthy data.")
        else:
            print("Warning: Healthy data is empty, skipping scaling.")

    # --- Split Healthy Data (Train/Validation) ---
    # We need the split *before* graph construction, as graph is built only on train data
    # X_val_healthy will now be used as the 'normal' portion of the final test set.
    X_train_healthy, X_val_healthy, y_train_healthy, y_val_healthy = train_test_split(
        healthy_sequences_full, healthy_labels_full, test_size=val_split, random_state=42, shuffle=True
    )
    print(f"Healthy Train set size: {len(X_train_healthy)}")
    print(f"Healthy Validation set size (used for thresholding & normal test samples): {len(X_val_healthy)}")

    # --- Graph Construction (using ONLY healthy TRAINING data) ---
    if len(X_train_healthy) == 0:
        raise ValueError("Healthy training data is empty. Cannot construct graph.")

    kmeans_model, node_features, adj_matrix = construct_graph_components(
        X_train_healthy, n_clusters, target_seq_len, input_dim
    )

    # --- Calculate State Indices for ALL splits using the fitted kmeans_model ---
    print("Calculating state indices for all datasets...")
    if len(X_train_healthy) > 0:
        train_entropies = np.array([calculate_entropy(seq) for seq in X_train_healthy])
        train_state_indices = np.array([get_window_state(e, kmeans_model) for e in train_entropies])
    else:
        train_state_indices = np.array([])

    # Calculate state indices for the validation split (healthy data)
    if len(X_val_healthy) > 0:
        val_entropies = np.array([calculate_entropy(seq) for seq in X_val_healthy])
        val_state_indices = np.array([get_window_state(e, kmeans_model) for e in val_entropies])
    else:
        val_state_indices = np.array([])

    # --- Load and Process Unhealthy (Anomalous) Test Data ---
    if test_data_type == 'high_energy':
        unhealthy_file = os.path.join(data_dir, "11_REFIT_B2_DW_unhealthy_high_energy_activations.npz")
    else:
        raise ValueError(f"Unsupported test_data_type: {test_data_type}")

    unhealthy_activations_list = load_npz_to_list(unhealthy_file)
    if not unhealthy_activations_list:
        print(f"Warning: Could not load unhealthy data. Test set will only contain normal samples.")
        X_test_unhealthy, y_test_unhealthy, test_state_indices_unhealthy = np.empty((0, target_seq_len, input_dim)), np.array([]), np.array([])
    else:
        X_test_raw_unhealthy = [act[:, appliance_col:appliance_col+1] for act in unhealthy_activations_list]
        X_test_processed_unhealthy = [_pad_or_truncate(seq, target_seq_len) for seq in X_test_raw_unhealthy]
        X_test_unhealthy = np.array(X_test_processed_unhealthy)
        y_test_unhealthy = np.ones(len(X_test_unhealthy)) # Label unhealthy data as 1

        # Scale Unhealthy Test Data
        if scale_data and scaler and X_test_unhealthy.size > 0:
            original_shape_test = X_test_unhealthy.shape
            data_to_scale_test = X_test_unhealthy.reshape(-1, original_shape_test[-1])
            scaled_data_test = scaler.transform(data_to_scale_test)
            X_test_unhealthy = scaled_data_test.reshape(original_shape_test)
            print("Applied MinMaxScaler to unhealthy test data.")

        # Calculate State Indices for Unhealthy Test Data
        if len(X_test_unhealthy) > 0:
            test_entropies_unhealthy = np.array([calculate_entropy(seq) for seq in X_test_unhealthy])
            test_state_indices_unhealthy = np.array([get_window_state(e, kmeans_model) for e in test_entropies_unhealthy])
        else:
            test_state_indices_unhealthy = np.array([])

    # --- Combine Healthy (Validation Split) and Unhealthy Data for FINAL Test Set ---
    X_test = np.concatenate((X_val_healthy, X_test_unhealthy), axis=0)
    y_test = np.concatenate((y_val_healthy, y_test_unhealthy), axis=0) # y_val_healthy are 0s, y_test_unhealthy are 1s
    test_state_indices = np.concatenate((val_state_indices, test_state_indices_unhealthy), axis=0)

    print(f"Combined Test set size: {len(X_test)} (Normal: {len(X_val_healthy)}, Anomalous: {len(X_test_unhealthy)})")

    # --- Create Datasets and DataLoaders ---
    train_dataset = DishwasherDatasetGCN(X_train_healthy, y_train_healthy, train_state_indices)
    # Validation dataset uses only the healthy validation split (for thresholding)
    val_dataset = DishwasherDatasetGCN(X_val_healthy, y_val_healthy, val_state_indices)
    # Test dataset uses the combined healthy + unhealthy data
    test_dataset = DishwasherDatasetGCN(X_test, y_test, test_state_indices)

    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers) # Batch size 1 for testing

    return (
        train_loader, val_loader, test_loader,
        scaler, kmeans_model, node_features, adj_matrix,
        target_seq_len, input_dim
    ) 