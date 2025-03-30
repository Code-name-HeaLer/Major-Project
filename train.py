import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

from model import GCNTransformerAutoencoder, GCNConv
from dataset import get_dataloaders_gcn
from utils import calculate_threshold_gcn, evaluate_model_gcn, plot_reconstructions


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap='Blues'):
    """Plots the confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_gcn.png')
    plt.close()

def plot_performance_metrics(accuracy, precision, recall, f1):
    """Plots the performance metrics as a bar chart."""
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 5))
    plt.bar(names, values, color=['blue', 'green', 'orange', 'red'])
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Performance Metrics (GCN-Transformer)")
    plt.ylim([0, 1]) # Scores are between 0 and 1
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', va='bottom')
    plt.savefig('performance_metrics_gcn.png')
    plt.close()

def train_gcn(model, train_loader, val_loader, criterion, optimizer, device,
            num_epochs, patience, node_features, adj_matrix):
    """Training loop for GCN-Transformer model with early stopping."""
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    # Move graph components to device once
    node_features = node_features.to(device)
    adj_matrix = adj_matrix.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        # Wrap train_loader with tqdm
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for i, batch in enumerate(train_iterator):
            seq = batch['sequence'].to(device)
            state_idx = batch['state_index'].to(device)

            optimizer.zero_grad()
            output = model(seq, state_idx, node_features, adj_matrix)
            loss = criterion(output, seq)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Update tqdm description with current loss
            train_iterator.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        # Wrap val_loader with tqdm
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for batch in val_iterator:
                seq = batch['sequence'].to(device)
                state_idx = batch['state_index'].to(device)
                output = model(seq, state_idx, node_features, adj_matrix)
                loss = criterion(output, seq)
                val_loss += loss.item()
                val_iterator.set_postfix({'val_loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        epoch_duration = time.time() - start_time
        # Tqdm leaves the last iteration progress bar, so print above it
        print(f'\nEpoch [{epoch+1}/{num_epochs}] completed in {epoch_duration:.2f}s, Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_gcn_transformer_autoencoder.pth')
            print(f'Validation loss decreased. Saving best model.')
        else:
            epochs_no_improve += 1
            print(f'Validation loss did not improve for {epochs_no_improve} epoch(s).')

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break

    print('Finished Training')
    # Plot training/validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs (GCN-Transformer)')
    plt.legend()
    plt.savefig('loss_plot_gcn.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GCN-Transformer Autoencoder for Anomaly Detection")
    parser.add_argument('--data_dir', type=str, default='dishwasher-dataset', help='Directory containing dataset files')
    parser.add_argument('--appliance_col', type=int, default=2, help='Index of the appliance energy column')
    parser.add_argument('--scale_data', type=bool, default=True, help='Apply MinMaxScaler to data')
    parser.add_argument('--test_data_type', type=str, default='high_energy', help='Type of unhealthy data for testing')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--d_model', type=int, default=64, help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of Transformer encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='Number of Transformer decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=128, help='Dimension of Transformer feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters/states for graph nodes')
    parser.add_argument('--gcn_hidden_dim', type=int, default=64, help='Hidden dimension for GCN layers')
    parser.add_argument('--gcn_out_dim', type=int, default=32, help='Output dimension for GCN state embeddings')
    parser.add_argument('--gcn_layers', type=int, default=8, help='Number of GCN layers (Note: 8 is deep, consider 2-4 if training is slow/unstable)')
    parser.add_argument('--threshold_percentile', type=int, default=95, help='Percentile for anomaly threshold')
    parser.add_argument('--plot_results', action='store_true', default=True, help='Plot confusion matrix and metrics')

    args = parser.parse_args()

    # --- Check for PyTorch Geometric ---
    if GCNConv is None:
        print("Error: PyTorch Geometric is required but not installed. Exiting.")
        exit(1)

    # --- Setup Device ---
    use_gpu = not args.no_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"Using device: {device}")
    if use_gpu:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # --- Load Data & Construct Graph ---
    print("Loading data and constructing graph...")
    train_loader, val_loader, test_loader, scaler, kmeans_model, node_features, adj_matrix, seq_len, input_dim = get_dataloaders_gcn(
        data_dir=args.data_dir,
        appliance_col=args.appliance_col,
        batch_size=args.batch_size,
        val_split=0.2,
        test_data_type=args.test_data_type,
        scale_data=args.scale_data,
        n_clusters=args.n_clusters
    )

    if seq_len is None:
        raise ValueError("Sequence length could not be determined.")
    if node_features is None or adj_matrix is None:
         raise ValueError("Graph construction failed. Check healthy training data and n_clusters.")

    # --- Initialize Model, Criterion, Optimizer ---
    print("Initializing GCN-Transformer model...")
    model = GCNTransformerAutoencoder(
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        n_clusters=args.n_clusters,
        gcn_hidden_dim=args.gcn_hidden_dim,
        gcn_out_dim=args.gcn_out_dim,
        gcn_layers=args.gcn_layers,
        dropout=args.dropout
    ).to(device)

    criterion = nn.L1Loss(reduction='mean') # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Train Model ---
    print("Starting training...")
    train_gcn(model, train_loader, val_loader, criterion, optimizer, device,
              args.epochs, args.patience, node_features, adj_matrix)

    # --- Load Best Model ---
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load('best_gcn_transformer_autoencoder.pth', map_location=device))

    # --- Calculate Threshold ---
    print("Calculating anomaly threshold...")
    threshold = calculate_threshold_gcn(model, val_loader, device, criterion,
                                       node_features, adj_matrix,
                                       percentile=args.threshold_percentile)

    # --- Evaluate Model ---
    print("Evaluating model on test set...")
    if len(test_loader.dataset) > 0:
        # Update evaluation call to get losses and sample reconstructions
        true_labels, predictions, test_losses, sample_recons_test = evaluate_model_gcn(
            model, test_loader, device, criterion, threshold,
            node_features, adj_matrix
        )

        # --- Report Metrics ---
        print("\nClassification Report:")
        target_names = ['Normal', 'Anomaly']
        try:
            print(classification_report(true_labels, predictions, target_names=target_names, zero_division=0))
        except ValueError as e:
            print(f"Could not generate classification report: {e}")
            print("This might happen if only one class was predicted across the test set.")
            print(f"True Labels sample: {true_labels[:10]}")
            print(f"Predictions sample: {predictions[:10]}")

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, pos_label=1, zero_division=0)
        recall = recall_score(true_labels, predictions, pos_label=1, zero_division=0)
        f1 = f1_score(true_labels, predictions, pos_label=1, zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Anomaly): {precision:.4f}")
        print(f"Recall (Anomaly): {recall:.4f}")
        print(f"F1 Score (Anomaly): {f1:.4f}")

        # --- Plot Results ---
        if args.plot_results:
            print("Plotting results...")
            try:
                cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
                plot_confusion_matrix(cm, classes=target_names, title='Confusion Matrix (GCN-Transformer)')
                plot_performance_metrics(accuracy, precision, recall, f1)
                # Call the new reconstruction plotting function
                plot_reconstructions(
                    val_loader, sample_recons_test, model, device,
                    node_features, adj_matrix, scaler # Pass scaler if used
                )
                print("Plots saved as confusion_matrix_gcn.png, performance_metrics_gcn.png, and reconstruction_examples_gcn.png")
            except Exception as e:
                print(f"Error during plotting: {e}")

    else:
        print("Test set is empty. Skipping evaluation and plotting.")

    print("Script finished.") 