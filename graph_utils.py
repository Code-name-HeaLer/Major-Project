import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy
import torch

def calculate_entropy(window: np.ndarray, base=2) -> float:
    """Calculates the entropy of a time series window."""
    # Ensure non-negative values and handle potential zeros for entropy calculation
    # Normalize the window to represent a distribution (sum to 1)
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-9
    window_positive = window - window.min() + epsilon # Make non-negative
    if window_positive.sum() == 0:
        return 0.0 # Entropy is 0 if all values are the same
    normalized_window = window_positive / window_positive.sum()
    return entropy(normalized_window.flatten(), base=base)

def construct_graph_components(sequences: np.ndarray, n_clusters: int, seq_len: int, input_dim: int):
    """
    Constructs the graph components (Nodes based on K-means clustering of window entropy,
    Node Features as mean windows, Adjacency Matrix based on Markov transitions).

    Args:
        sequences: Numpy array of healthy training sequences [num_samples, seq_len, input_dim].
        n_clusters: Number of states/nodes (K in K-means).
        seq_len: Length of each sequence window.
        input_dim: Feature dimension of each sequence window.

    Returns:
        kmeans_model: Fitted K-means model.
        node_features: Tensor of node features (mean windows) [n_clusters, seq_len, input_dim].
        adj_matrix: Tensor of the weighted adjacency matrix [n_clusters, n_clusters].
    """
    print(f"Starting graph construction with {n_clusters} clusters...")

    # 1. Calculate Entropy for each sequence
    print("Calculating entropy for windows...")
    sequence_entropies = np.array([calculate_entropy(seq) for seq in sequences])
    sequence_entropies = sequence_entropies.reshape(-1, 1) # Reshape for KMeans

    # 2. Cluster windows based on entropy using K-means
    print("Clustering windows using K-means based on entropy...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Use n_init
    cluster_labels = kmeans.fit_predict(sequence_entropies)
    print(f"K-means clustering complete. Found {len(np.unique(cluster_labels))} unique clusters.")
    if len(np.unique(cluster_labels)) < n_clusters:
         print(f"Warning: K-means found fewer clusters ({len(np.unique(cluster_labels))}) than requested ({n_clusters}). Some states might be empty.")
         # Consider re-running k-means or using a different clustering approach if this is problematic.

    # 3. Initialize Node Features (Mean of assigned windows)
    print("Calculating node features (mean windows per cluster)...")
    node_features_list = []
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) > 0:
            mean_window = np.mean(sequences[cluster_indices], axis=0)
            node_features_list.append(mean_window)
        else:
            # Handle empty clusters: Initialize with zeros or mean of all data
            print(f"Warning: Cluster {i} is empty. Initializing node feature with zeros.")
            node_features_list.append(np.zeros((seq_len, input_dim)))

    node_features = torch.tensor(np.array(node_features_list), dtype=torch.float32)
    # Shape: [n_clusters, seq_len, input_dim]

    # 4. Calculate Transition Probabilities (Markov Chain)
    print("Calculating transition probabilities (adjacency matrix)...")
    transition_counts = np.zeros((n_clusters, n_clusters))
    for i in range(len(cluster_labels) - 1):
        current_state = cluster_labels[i]
        next_state = cluster_labels[i+1]
        transition_counts[current_state, next_state] += 1

    # Normalize counts to get probabilities (add epsilon for stability)
    epsilon_adj = 1e-6
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    adj_matrix_np = transition_counts / (row_sums + epsilon_adj)

    # Handle rows with zero sum (states with no outgoing transitions)
    zero_sum_rows = np.where(row_sums == 0)[0]
    if len(zero_sum_rows) > 0:
        print(f"Warning: States {zero_sum_rows} have no outgoing transitions. Setting self-loops to 1.")
        for row_idx in zero_sum_rows:
            adj_matrix_np[row_idx, row_idx] = 1.0 # Assign self-loop probability of 1

    adj_matrix = torch.tensor(adj_matrix_np, dtype=torch.float32)
    # Shape: [n_clusters, n_clusters]

    print("Graph construction complete.")
    return kmeans, node_features, adj_matrix


def get_window_state(window_entropy: float, kmeans_model) -> int:
    """Predicts the cluster/state for a given window's entropy using the trained K-means model."""
    # Reshape the single entropy value
    entropy_array = np.array([[window_entropy]])
    state = kmeans_model.predict(entropy_array)[0]
    return state 