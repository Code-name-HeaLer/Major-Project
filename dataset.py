import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings # Import warnings

from utils import load_npz_to_list
# Import GCNConv directly from model to check availability
try:
    from model import GCNConv # Changed from GCNTransformerAutoencoder
except ImportError:
    GCNConv = None # Set to None if import fails

if GCNConv is not None: # Check the imported variable directly
    from graph_utils import construct_graph_components, calculate_entropy, get_window_state
else:
    # Define dummy functions if torch_geometric is not available
    print("Defining dummy graph functions as PyTorch Geometric is not installed.")
    def calculate_entropy(*args, **kwargs):
        return 0.0
    def construct_graph_components(*args, **kwargs):
        # Return expected number of values, even if None
        return None, None, None # Removed kmeans_model from dummy return
    def get_window_state(*args, **kwargs):
        return 0

class DishwasherDatasetGCN(Dataset):
    """PyTorch Dataset including sequence state index and anomaly type."""
    # Added anomaly_type to init and storage
    def __init__(self, sequences, labels, state_indices, anomaly_types):
        self.sequences = sequences
        self.labels = labels
        self.state_indices = state_indices
        self.anomaly_types = anomaly_types # Store anomaly types

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        state_index = self.state_indices[idx]
        anomaly_type = self.anomaly_types[idx] # Get anomaly type
        # Ensure sequence is float32 and state_index is long
        return {
            'sequence': torch.tensor(sequence, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'state_index': torch.tensor(state_index, dtype=torch.long),
            'anomaly_type': anomaly_type # Return anomaly type
        }

def _pad_or_truncate(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Pads or truncates a sequence to the target length."""
    current_len = seq.shape[0]
    # Handle cases where seq might be None or empty after loading
    if seq is None or seq.shape[0] == 0:
         # Return array of zeros with correct shape (target_len, num_features)
         # We need the number of features. Assuming it's 1 based on previous context.
         # A more robust way would be to pass expected num_features.
         # For now, let's handle the case where seq has shape info even if empty.
         num_features = seq.shape[1] if len(seq.shape) > 1 else 1
         warnings.warn(f"Attempting to pad/truncate an empty or None sequence. Returning zeros shape=({target_len}, {num_features}).")
         return np.zeros((target_len, num_features), dtype=np.float32) # Use float32

    num_features = seq.shape[1]
    if current_len == target_len:
        return seq
    elif current_len > target_len:
        return seq[:target_len, :]
    else:
        padding_size = target_len - current_len
        # Ensure padding matches the sequence's feature dimension
        padding = np.zeros((padding_size, num_features), dtype=seq.dtype)
        return np.vstack((seq, padding))

def get_dataloaders_gcn(data_dir: str, appliance_col: int = 2, seq_len: int = None,
                        batch_size: int = 32, val_split: float = 0.2, # val_split for thresholding
                        scale_data: bool = True, n_clusters: int = 10, fixed_seq_len: int = None):
    """
    Loads data, performs graph construction on training data, preprocesses, splits,
    calculates state indices, and creates DataLoaders for the GCN-Transformer model.
    Test set includes ALL healthy and ALL specified unhealthy data.

    Args:
        data_dir: Directory containing the .npz files.
        appliance_col: Index of the column containing appliance energy data (default: 2).
        seq_len: Target sequence length. If None, determines max length from healthy data.
        batch_size: Batch size for DataLoaders.
        val_split: Fraction of *healthy* data to reserve for validation (used for threshold calculation).
        scale_data: Whether to apply MinMaxScaler to the appliance energy data.
        n_clusters: Number of states/nodes for graph construction.
        fixed_seq_len: Fixed target sequence length. If provided, overrides seq_len.

    Returns:
        train_loader: DataLoader for healthy training data.
        val_loader: DataLoader for healthy validation data (used for thresholding).
        test_loader: DataLoader for the combined test set (all healthy + all unhealthy).
        scaler: Fitted MinMaxScaler or None.
        kmeans_model: Fitted K-means model from graph construction.
        node_features: Tensor of node features (mean windows).
        adj_matrix: Tensor of the weighted adjacency matrix.
        target_seq_len: The sequence length used.
        input_dim: Feature dimension (usually 1).
        all_sequences: Dictionary holding all loaded sequences before splitting {'healthy': [...], 'unhealthy': {'type1': [...], ...}}
        all_labels: Dictionary holding corresponding labels {'healthy': [...], 'unhealthy': {'type1': [...], ...}}
        all_anomaly_types: Dictionary holding anomaly type strings {'healthy': [...], 'unhealthy': {'type1': [...], ...}}

    """
    if GCNConv is None: # Check the imported variable directly
        raise ImportError("PyTorch Geometric required for GCN data loading.")

    print("--- Starting Data Loading and Preprocessing ---")

    # --- Load ALL Healthy Data ---
    healthy_file = os.path.join(data_dir, "11_REFIT_B2_DW_healthy_activations.npz")
    healthy_activations_list = load_npz_to_list(healthy_file)
    if not healthy_activations_list:
        raise ValueError(f"Could not load healthy data from {healthy_file}")
    print(f"Loaded {len(healthy_activations_list)} healthy activations.")

    # --- Load ALL Unhealthy Data ---
    unhealthy_files = {
        "high_energy": os.path.join(data_dir, "11_REFIT_B2_DW_unhealthy_high_energy_activations.npz"),
        "repeated_cycle": os.path.join(data_dir, "11_REFIT_B2_DW_unhealthy_repeated_cycle_activations.npz"),
        "noisy_cycle": os.path.join(data_dir, "11_REFIT_B2_DW_unhealthy_noisy_activations.npz")
    }
    unhealthy_activations_map = {}
    total_unhealthy_count = 0
    for anomaly_type, filepath in unhealthy_files.items():
        # Use a try-except block specifically for FileNotFoundError during loading
        try:
            activations = load_npz_to_list(filepath) # load_npz_to_list should handle FileNotFoundError internally now
            if not activations: # Check if load_npz_to_list returned empty list due to error or file not found
                 warnings.warn(f"Could not load or file not found for unhealthy data type '{anomaly_type}' from {filepath}. Skipping.")
                 unhealthy_activations_map[anomaly_type] = []
            else:
                print(f"Loaded {len(activations)} unhealthy '{anomaly_type}' activations.")
                unhealthy_activations_map[anomaly_type] = activations
                total_unhealthy_count += len(activations)
        except Exception as e: # Catch any other unexpected loading errors
             warnings.warn(f"An unexpected error occurred loading '{anomaly_type}' from {filepath}: {e}. Skipping.")
             unhealthy_activations_map[anomaly_type] = []

    print(f"Total unhealthy activations loaded: {total_unhealthy_count}")

    # --- Extract Appliance Column and Determine Sequence Length ---
    all_sequences_raw_map = {'healthy': [], 'unhealthy': {}}
    all_lengths = []

    # Process healthy
    all_sequences_raw_map['healthy'] = [act[:, appliance_col:appliance_col+1] for act in healthy_activations_list if act is not None and act.shape[0] > 0]
    all_lengths.extend([s.shape[0] for s in all_sequences_raw_map['healthy']])

    # Process unhealthy
    for anomaly_type, activations in unhealthy_activations_map.items():
         # Ensure list comprehension handles empty activation lists gracefully
         all_sequences_raw_map['unhealthy'][anomaly_type] = [act[:, appliance_col:appliance_col+1] for act in activations if act is not None and act.shape[0] > 0]
         all_lengths.extend([s.shape[0] for s in all_sequences_raw_map['unhealthy'][anomaly_type]])

    if not all_lengths and fixed_seq_len is None: # Check if lengths list is empty AND no fixed length provided
        raise ValueError("No valid sequences found and no fixed_seq_len provided.")

    # Determine target sequence length
    # PRIORITIZE fixed_seq_len if provided
    if fixed_seq_len is not None:
        target_seq_len = fixed_seq_len
        print(f"Using provided fixed target sequence length: {target_seq_len}")
    elif not seq_len and all_lengths: # Original logic: if no specific seq_len arg AND we have lengths
        target_seq_len = max(all_lengths)
        print(f"Determined target sequence length from all data (max): {target_seq_len}")
    elif seq_len: # Original logic: if seq_len arg was provided (but fixed_seq_len takes precedence now)
        target_seq_len = seq_len
        print(f"Using specified target sequence length: {target_seq_len}")
    else: # Should not happen if the check above worked, but as a fallback
         raise ValueError("Could not determine target sequence length.")

    # Check input dimension consistency (should be 1)
    first_seq = next((s for s in all_sequences_raw_map['healthy'] if s is not None), None)
    if first_seq is None: # If no healthy, check unhealthy
         first_seq = next((s for atype in all_sequences_raw_map['unhealthy']
                            for s in all_sequences_raw_map['unhealthy'][atype] if s is not None), None)
    if first_seq is None:
         raise ValueError("Could not determine input dimension, no valid sequences found.")
    input_dim = first_seq.shape[1]
    print(f"Input dimension: {input_dim}")

    # --- Pad/Truncate ALL Sequences ---
    all_sequences_processed = {'healthy': [], 'unhealthy': {}}
    all_labels = {'healthy': [], 'unhealthy': {}}
    all_anomaly_types = {'healthy': [], 'unhealthy': {}} # Store type strings

    # Healthy
    # Handle case where healthy list might be empty after filtering
    if all_sequences_raw_map['healthy']:
        all_sequences_processed['healthy'] = np.array([_pad_or_truncate(seq, target_seq_len) for seq in all_sequences_raw_map['healthy']])
        num_healthy = len(all_sequences_processed['healthy'])
        all_labels['healthy'] = np.zeros(num_healthy)
        all_anomaly_types['healthy'] = ['normal'] * num_healthy
    else: # Ensure these exist as empty structures if no healthy data
        all_sequences_processed['healthy'] = np.empty((0, target_seq_len, input_dim), dtype=np.float32)
        all_labels['healthy'] = np.array([])
        all_anomaly_types['healthy'] = []
        num_healthy = 0

    # Unhealthy
    for anomaly_type, sequences_raw in all_sequences_raw_map['unhealthy'].items():
        if sequences_raw: # Check if list is not empty
            processed_seqs = np.array([_pad_or_truncate(seq, target_seq_len) for seq in sequences_raw])
            all_sequences_processed['unhealthy'][anomaly_type] = processed_seqs
            num_unhealthy_type = len(processed_seqs)
            all_labels['unhealthy'][anomaly_type] = np.ones(num_unhealthy_type)
            all_anomaly_types['unhealthy'][anomaly_type] = [anomaly_type] * num_unhealthy_type
        else: # Ensure these exist as empty structures if no data for this type
             all_sequences_processed['unhealthy'][anomaly_type] = np.empty((0, target_seq_len, input_dim), dtype=np.float32)
             all_labels['unhealthy'][anomaly_type] = np.array([])
             all_anomaly_types['unhealthy'][anomaly_type] = []


    # Combine all sequences into one large array for scaling
    all_seqs_list = [all_sequences_processed['healthy']]
    # Only extend with non-empty arrays
    all_seqs_list.extend([arr for arr in all_sequences_processed['unhealthy'].values() if arr.size > 0])
    # Filter out empty arrays before concatenating (double check)
    all_seqs_list_filtered = [arr for arr in all_seqs_list if arr.size > 0]

    if not all_seqs_list_filtered:
        # If still no sequences, something is wrong (maybe only empty files loaded)
        raise ValueError("No sequences available after processing and padding healthy and unhealthy data.")

    combined_sequences = np.concatenate(all_seqs_list_filtered, axis=0)

    # --- Scale Data (Optional) ---
    # Fit scaler ONLY on healthy data, then transform all data.
    scaler = None
    if scale_data:
        healthy_data_for_scaling = all_sequences_processed['healthy']
        if healthy_data_for_scaling.size > 0:
            original_shape_healthy = healthy_data_for_scaling.shape
            # Ensure healthy data is 2D for scaler
            data_to_scale_healthy = healthy_data_for_scaling.reshape(-1, input_dim)

            scaler = MinMaxScaler()
            print("Fitting MinMaxScaler ONLY on healthy data...")
            scaler.fit(data_to_scale_healthy)

            print("Applying MinMaxScaler to ALL data...")
            original_shape_all = combined_sequences.shape
            # Ensure combined data is 2D for scaler transform
            combined_sequences_flat = combined_sequences.reshape(-1, input_dim)
            scaled_data_all = scaler.transform(combined_sequences_flat)
            combined_sequences = scaled_data_all.reshape(original_shape_all)

            # Update the processed sequences map with scaled data
            start_idx = 0
            # Healthy
            if all_sequences_processed['healthy'].size > 0:
                end_idx = start_idx + len(all_sequences_processed['healthy'])
                all_sequences_processed['healthy'] = combined_sequences[start_idx:end_idx]
                start_idx = end_idx
            # Unhealthy
            for anomaly_type in all_sequences_processed['unhealthy']:
                 num_seqs = len(all_sequences_processed['unhealthy'][anomaly_type])
                 if num_seqs > 0:
                      end_idx = start_idx + num_seqs
                      all_sequences_processed['unhealthy'][anomaly_type] = combined_sequences[start_idx:end_idx]
                      start_idx = end_idx

            print("Applied MinMaxScaler to all data using scaler fitted on healthy data.")
        else:
            warnings.warn("Healthy data is empty, skipping scaling.")

    # --- Split Healthy Data (Train/Validation for training and thresholding) ---
    X_healthy = all_sequences_processed['healthy']
    y_healthy = all_labels['healthy']

    if len(X_healthy) == 0:
         # If no healthy data after all processing, cannot train/validate
         warnings.warn("No healthy data available after processing. Train/Validation loaders will be empty.")
         X_train_healthy, X_val_healthy = np.array([]), np.array([])
         y_train_healthy, y_val_healthy = np.array([]), np.array([])
    else:
        # Ensure val_split results in at least one sample for validation if possible
        if val_split > 0 and int(len(X_healthy) * val_split) == 0 and len(X_healthy) > 1:
            # If split is too small but > 0, assign at least one sample to validation
            num_val_samples = 1
            print(f"Validation split {val_split:.2f} resulted in 0 samples for {len(X_healthy)} healthy items. Setting validation size to {num_val_samples}.")
        elif val_split <= 0:
             num_val_samples = 0
             print("Validation split is <= 0. Validation set will be empty.")
        else:
            num_val_samples = int(len(X_healthy) * val_split)


        if num_val_samples == 0:
             # Use all healthy for training, none for validation
             X_train_healthy = X_healthy
             y_train_healthy = y_healthy
             X_val_healthy = np.empty((0, target_seq_len, input_dim), dtype=np.float32)
             y_val_healthy = np.array([])
             if val_split > 0: # Only warn if user intended a split
                  warnings.warn("Validation set is empty (split resulted in 0 samples or val_split=0). Thresholding might be based on training data or default.")
        elif num_val_samples >= len(X_healthy):
            # Use all healthy for validation, none for training
            X_train_healthy = np.empty((0, target_seq_len, input_dim), dtype=np.float32)
            y_train_healthy = np.array([])
            X_val_healthy = X_healthy
            y_val_healthy = y_healthy
            warnings.warn("Validation split is >= 1.0. Using all healthy data for validation. Training set will be empty.")
        else:
            X_train_healthy, X_val_healthy, y_train_healthy, y_val_healthy = train_test_split(
                X_healthy, y_healthy, test_size=num_val_samples, random_state=42, shuffle=True
            )

    print(f"Healthy Train set size: {len(X_train_healthy)}")
    print(f"Healthy Validation set size (for thresholding): {len(X_val_healthy)}")

    # --- Graph Construction (using ONLY healthy TRAINING data) ---
    kmeans_model, node_features, adj_matrix = None, None, None
    if len(X_train_healthy) > 0:
        print("--- Starting Graph Construction (on healthy train data) ---")
        try:
            # Pass input_dim to construct_graph_components if needed
            kmeans_model, node_features, adj_matrix = construct_graph_components(
                X_train_healthy, n_clusters, target_seq_len, input_dim
            )
            if kmeans_model is None or node_features is None or adj_matrix is None:
                 warnings.warn("Graph construction did not return valid components. Check graph_utils.py. Proceeding without graph features.")
        except Exception as e:
             warnings.warn(f"Error during graph construction: {e}. Proceeding without graph features.")

    else:
        warnings.warn("Healthy training data is empty. Skipping graph construction.")


    # --- Prepare Combined Test Set (ALL Healthy + ALL Unhealthy) ---
    # Use the already processed and potentially scaled data
    X_test_list = [all_sequences_processed['healthy']]
    y_test_list = [all_labels['healthy']]
    anomaly_types_test_list = [all_anomaly_types['healthy']]

    for anomaly_type in all_sequences_processed['unhealthy']:
        # Only include if data exists for this type
        if all_sequences_processed['unhealthy'][anomaly_type].size > 0:
            X_test_list.append(all_sequences_processed['unhealthy'][anomaly_type])
            y_test_list.append(all_labels['unhealthy'][anomaly_type])
            anomaly_types_test_list.append(all_anomaly_types['unhealthy'][anomaly_type])

    # Filter empty arrays again before concatenating (safety check)
    X_test_list_filt = [arr for arr in X_test_list if arr.size > 0]
    y_test_list_filt = [arr for arr in y_test_list if arr.size > 0]
    anomaly_types_test_list_filt = [lst for lst in anomaly_types_test_list if len(lst) > 0]


    if not X_test_list_filt:
         # If test set is empty even after combining, raise error or return empty loaders
         warnings.warn("Test data is empty after combining healthy and unhealthy samples. Test loader will be empty.")
         X_test = np.empty((0, target_seq_len, input_dim), dtype=np.float32)
         y_test = np.array([])
         test_anomaly_types = []
    else:
        X_test = np.concatenate(X_test_list_filt, axis=0)
        y_test = np.concatenate(y_test_list_filt, axis=0)
        # Flatten list of lists for anomaly types
        test_anomaly_types = [item for sublist in anomaly_types_test_list_filt for item in sublist]

    print(f"Combined Test set size: {len(X_test)}")
    if len(X_test) > 0:
        unique_labels, counts = np.unique(y_test, return_counts=True)
        print(f"Test set composition: {dict(zip(unique_labels, counts))} (0: Normal, 1: Anomaly)")


    # --- Calculate State Indices for ALL splits using the fitted kmeans_model ---
    print("Calculating state indices for all datasets...")
    if kmeans_model is None:
        warnings.warn("KMeans model not available. Assigning default state index 0 to all sequences.")
        # Assign default state index 0 if kmeans failed
        train_state_indices = np.zeros(len(X_train_healthy), dtype=int)
        val_state_indices = np.zeros(len(X_val_healthy), dtype=int)
        test_state_indices = np.zeros(len(X_test), dtype=int)
    else:
        # Train
        if len(X_train_healthy) > 0:
            try:
                train_entropies = np.array([calculate_entropy(seq) for seq in X_train_healthy])
                train_state_indices = kmeans_model.predict(train_entropies.reshape(-1, 1)) # Use predict directly
            except Exception as e:
                 warnings.warn(f"Error calculating train state indices: {e}. Assigning default 0.")
                 train_state_indices = np.zeros(len(X_train_healthy), dtype=int)
        else:
            train_state_indices = np.array([], dtype=int)
        # Validation
        if len(X_val_healthy) > 0:
             try:
                val_entropies = np.array([calculate_entropy(seq) for seq in X_val_healthy])
                val_state_indices = kmeans_model.predict(val_entropies.reshape(-1, 1))
             except Exception as e:
                 warnings.warn(f"Error calculating validation state indices: {e}. Assigning default 0.")
                 val_state_indices = np.zeros(len(X_val_healthy), dtype=int)
        else:
            val_state_indices = np.array([], dtype=int)
        # Test
        if len(X_test) > 0:
            try:
                test_entropies = np.array([calculate_entropy(seq) for seq in X_test])
                test_state_indices = kmeans_model.predict(test_entropies.reshape(-1, 1))
            except Exception as e:
                 warnings.warn(f"Error calculating test state indices: {e}. Assigning default 0.")
                 test_state_indices = np.zeros(len(X_test), dtype=int)
        else:
            test_state_indices = np.array([], dtype=int)

    # --- Create Datasets and DataLoaders ---
    # Pass anomaly types to the Dataset constructor
    # Need anomaly types for train/val splits (even though they are all 'normal')
    train_anomaly_types = ['normal'] * len(X_train_healthy)
    val_anomaly_types = ['normal'] * len(X_val_healthy)

    # Handle empty datasets gracefully when creating DataLoaders
    train_dataset = DishwasherDatasetGCN(X_train_healthy, y_train_healthy, train_state_indices, train_anomaly_types) if len(X_train_healthy) > 0 else None
    val_dataset = DishwasherDatasetGCN(X_val_healthy, y_val_healthy, val_state_indices, val_anomaly_types) if len(X_val_healthy) > 0 else None
    test_dataset = DishwasherDatasetGCN(X_test, y_test, test_state_indices, test_anomaly_types) if len(X_test) > 0 else None

    num_workers = 0 # For stability, especially on Windows

    # Create loaders only if datasets exist
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True) if train_dataset else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False) if test_dataset else None # Batch size 1 for testing

    print("--- DataLoaders Created ---")

    # Prepare the dictionaries of all processed data for returning (used for plotting)
    all_sequences_dict = {
        'healthy': all_sequences_processed['healthy'],
        'unhealthy': all_sequences_processed['unhealthy'] # This now contains {type: array, ...}
    }
    all_labels_dict = {
        'healthy': all_labels['healthy'],
        'unhealthy': all_labels['unhealthy'] # This now contains {type: array, ...}
    }
    all_anomaly_types_dict = {
        'healthy': all_anomaly_types['healthy'],
        'unhealthy': all_anomaly_types['unhealthy'] # This now contains {type: list, ...}
    }


    return (
        train_loader, val_loader, test_loader,
        scaler, kmeans_model, node_features, adj_matrix,
        target_seq_len, input_dim,
        all_sequences_dict, all_labels_dict, all_anomaly_types_dict # Return all data for plotting if needed
    ) 