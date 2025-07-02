import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
import numpy as np
from typing import List, Tuple, Dict, Optional

class CustomSAGEConv(SAGEConv):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super().__init__(in_channels, out_channels)
        self.edge_lin = nn.Linear(edge_dim, in_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Edge özelliklerini dönüştür
        edge_attr = self.edge_lin(edge_attr)
        
        # Self-loops ekle
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
        
        # Normal SAGEConv forward
        return super().forward(x, edge_index)

class TemporalGraphSAGE(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_type_feat_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_edge_types: int = 249  # Dataset A'daki edge type sayısı
    ):
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_type_feat_dim = edge_type_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node feature embedding
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        
        # Edge type embedding
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(CustomSAGEConv(hidden_dim, hidden_dim, hidden_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor
    ) -> torch.Tensor:
        # Node feature embedding
        x = self.node_embedding(x)
        
        # Edge type embedding
        edge_emb = self.edge_type_embedding(edge_type)
        
        # GraphSAGE layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_emb)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Get source and destination node embeddings
        src_emb = x[src_idx]
        dst_emb = x[dst_idx]
        
        # Concatenate embeddings for link prediction
        combined = torch.cat([src_emb, dst_emb, src_emb * dst_emb, torch.abs(src_emb - dst_emb)], dim=1)
        
        # Predict link probability
        return self.link_predictor(combined).squeeze()

def prepare_temporal_batch(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    timestamps: torch.Tensor,
    time_window: Tuple[float, float]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Belirli bir zaman penceresindeki edge'leri filtreler.
    """
    mask = (timestamps >= time_window[0]) & (timestamps < time_window[1])
    return edge_index[:, mask], edge_type[mask], timestamps[mask]

def negative_sampling(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: int
) -> torch.Tensor:
    """
    Negatif örnekler oluşturur.
    """
    pos_edges = set(map(tuple, edge_index.t().numpy()))
    neg_edges = []
    
    while len(neg_edges) < num_neg_samples:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst and (src, dst) not in pos_edges:
            neg_edges.append([src, dst])
    
    return torch.tensor(neg_edges, dtype=torch.long).t() 