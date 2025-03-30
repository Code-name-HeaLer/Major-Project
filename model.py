import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PositionalEncoding
import math

# Check if torch_geometric is installed, provide informative error if not
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import dense_to_sparse
except ImportError:
    GCNConv = None
    dense_to_sparse = None
    print("Warning: PyTorch Geometric not found. GCN functionality will be disabled.")
    print("Please install it following instructions at: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")

class GCNEncoder(nn.Module):
    """Graph Convolutional Network Encoder."""
    def __init__(self, node_feature_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if GCNConv is None:
            raise ImportError("PyTorch Geometric is required for GCNEncoder.")

        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Graph connectivity in COO format [2, num_edges]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1: # Apply activation and dropout to all but last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        # Last layer output (as per readme: sigmoid activation)
        # Let's use ReLU for consistency with the original readme unless sigmoid is critical
        # Using Sigmoid for the GCN output might compress the range too much.
        # The readme mentioned sigmoid for the *last layer* of GCN, let's try that.
        x = torch.sigmoid(x)
        return x

class GCNTransformerAutoencoder(nn.Module):
    """Transformer Autoencoder integrated with GCN state embeddings."""
    def __init__(self, input_dim: int, seq_len: int, d_model: int, nhead: int,
                 num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int,
                 n_clusters: int, gcn_hidden_dim: int, gcn_out_dim: int, gcn_layers: int = 2, # Readme specified 8 GCN layers
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.gcn_out_dim = gcn_out_dim

        if GCNConv is None:
            raise ImportError("PyTorch Geometric is required for GCNTransformerAutoencoder.")

        # --- GCN Encoder ---
        # GCN expects node features as [num_nodes, features_per_node]
        # Our node features are mean windows [n_clusters, seq_len, input_dim]
        # We need to flatten or summarize the seq_len dimension for standard GCNConv
        # Option 1: Flatten -> GCN expects [n_clusters, seq_len * input_dim]
        gcn_node_feature_dim = seq_len * input_dim
        # Option 2: Use mean/max pooling over seq_len -> GCN expects [n_clusters, input_dim]
        # Let's use Flatten for now, as it preserves more info
        self.gcn_encoder = GCNEncoder(gcn_node_feature_dim, gcn_hidden_dim, gcn_out_dim,
                                      num_layers=gcn_layers, dropout=dropout)

        # --- Transformer Components ---
        # Input embedding layer for the sequence
        self.input_embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 1)

        # Transformer Encoder for the sequence
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Transformer Decoder
        # The memory input to the decoder will be concatenation of sequence encoding and GCN state embedding
        decoder_input_dim = d_model + gcn_out_dim
        # We need a linear layer to project decoder_input_dim -> d_model if decoder expects d_model memory
        # Alternatively, adjust decoder layer? Let's project the concatenated memory.
        self.memory_projection = nn.Linear(decoder_input_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, activation='relu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # --- Output Layer ---
        # Linear layer to reconstruct the sequence
        self.output_linear = nn.Linear(d_model, input_dim)
        # TODO: Add 1D Deconvolution if needed, replacing/augmenting self.output_linear

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_embed.weight.data.uniform_(-initrange, initrange)
        self.output_linear.bias.data.zero_()
        self.output_linear.weight.data.uniform_(-initrange, initrange)
        self.memory_projection.bias.data.zero_()
        self.memory_projection.weight.data.uniform_(-initrange, initrange)
        # Initialize GCN weights? GCNConv uses default initializations (Glorot)

    def forward(self, src: torch.Tensor, state_indices: torch.Tensor, # Input sequence and corresponding state index
                node_features: torch.Tensor, adj_matrix: torch.Tensor,
                tgt: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Input sequence windows, shape [batch_size, seq_len, input_dim]
            state_indices: Index of the state (cluster) for each window in the batch, shape [batch_size]
            node_features: Mean window for each state (cluster), shape [n_clusters, seq_len, input_dim]
            adj_matrix: Weighted adjacency matrix, shape [n_clusters, n_clusters]
            tgt: Target sequence for decoder (e.g., shifted src), shape [batch_size, seq_len, input_dim]

        Returns:
            output: Reconstructed sequence, shape [batch_size, seq_len, input_dim]
        """
        batch_size = src.size(0)
        device = src.device

        if dense_to_sparse is None:
             raise ImportError("PyTorch Geometric (dense_to_sparse) is required.")

        # --- Prepare Graph Data ---
        # GCN expects node features [num_nodes, features]
        gcn_node_features = node_features.view(node_features.size(0), -1).to(device) # Flatten seq_len * input_dim
        # Convert adjacency matrix to edge_index format [2, num_edges]
        edge_index, edge_weight = dense_to_sparse(adj_matrix.to(device))
        # Note: GCNConv doesn't directly use edge_weight by default, but it's good practice to have it.

        # --- GCN Encoder --- Pass node features and edge index
        # Output shape: [n_clusters, gcn_out_dim]
        all_state_embeddings = self.gcn_encoder(gcn_node_features, edge_index)

        # Select the embedding corresponding to the state of each window in the batch
        # Shape: [batch_size, gcn_out_dim]
        batch_state_embeddings = all_state_embeddings[state_indices]

        # --- Transformer Sequence Encoder ---
        if tgt is None:
            tgt = src # Use src as target for autoencoding

        src_embedded = self.input_embed(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoder(src_embedded.permute(1, 0, 2)).permute(1, 0, 2)
        # Output shape: [batch_size, seq_len, d_model]
        sequence_memory = self.transformer_encoder(src_embedded)

        # --- Combine Sequence and State Info for Decoder Memory ---
        # Expand state embedding to match sequence length for concatenation
        # Shape: [batch_size, seq_len, gcn_out_dim]
        expanded_state_embeddings = batch_state_embeddings.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Concatenate along the feature dimension
        # Shape: [batch_size, seq_len, d_model + gcn_out_dim]
        combined_memory = torch.cat((sequence_memory, expanded_state_embeddings), dim=-1)

        # Project combined memory back to d_model for the decoder
        # Shape: [batch_size, seq_len, d_model]
        projected_memory = self.memory_projection(combined_memory)

        # --- Transformer Decoder ---
        tgt_embedded = self.input_embed(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoder(tgt_embedded.permute(1, 0, 2)).permute(1, 0, 2)

        # Generate causal mask for target sequence
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)

        # Decoder processes target embedding and projected combined memory
        # Output shape: [batch_size, seq_len, d_model]
        output = self.transformer_decoder(tgt_embedded, projected_memory, tgt_mask=tgt_mask)

        # --- Output Layer ---
        # Shape: [batch_size, seq_len, input_dim]
        output = self.output_linear(output)

        return output 