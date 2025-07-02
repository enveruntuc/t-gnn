import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import LabelEncoder

class TemporalGraphDataLoader:
    def __init__(
        self,
        edges_path: str,
        node_features_path: str,
        edge_type_features_path: str,
        input_path: str
    ):
        self.edges_path = edges_path
        self.node_features_path = node_features_path
        self.edge_type_features_path = edge_type_features_path
        self.input_path = input_path
        
        # Önce dosyaları okuyup sütun sayılarını belirle
        edges_temp = pd.read_csv(edges_path, header=None)
        node_features_temp = pd.read_csv(node_features_path, header=None)
        edge_type_features_temp = pd.read_csv(edge_type_features_path, header=None)
        input_temp = pd.read_csv(input_path, header=None)
        
        # Sütun sayılarını al
        num_node_features = node_features_temp.shape[1] - 1  # node_id hariç
        num_edge_type_features = edge_type_features_temp.shape[1] - 1  # edge_type hariç
        
        # Veri yükleme - sütun isimlerini belirterek
        self.edges_df = pd.read_csv(
            edges_path,
            header=None,
            names=['src_id', 'dst_id', 'edge_type', 'timestamp']
        )
        
        self.node_features_df = pd.read_csv(
            node_features_path,
            header=None,
            names=['node_id'] + [f'feature_{i}' for i in range(1, num_node_features + 1)]
        )
        
        self.edge_type_features_df = pd.read_csv(
            edge_type_features_path,
            header=None,
            names=['edge_type'] + [f'feature_{i}' for i in range(1, num_edge_type_features + 1)]
        )
        
        self.input_df = pd.read_csv(
            input_path,
            header=None,
            names=['src_id', 'dst_id', 'edge_type', 'start_time', 'end_time']
        )
        
        # Label encoding
        self.node_encoder = LabelEncoder()
        self.edge_type_encoder = LabelEncoder()
        
        # Veri ön işleme
        self._preprocess_data()
        
        print(f"Veri yüklendi:")
        print(f"- Node özellik sayısı: {num_node_features}")
        print(f"- Edge type özellik sayısı: {num_edge_type_features}")
        print(f"- Toplam düğüm sayısı: {len(self.node_encoder.classes_)}")
        print(f"- Toplam kenar tipi sayısı: {len(self.edge_type_encoder.classes_)}")
        print(f"- Toplam kenar sayısı: {len(self.edges_df)}")
        print(f"- Tahmin edilecek örnek sayısı: {len(self.input_df)}")
    
    def _preprocess_data(self):
        """Veriyi ön işler ve tensörlere dönüştürür."""
        # Node ID'leri encode et
        all_nodes = pd.concat([
            self.edges_df['src_id'],
            self.edges_df['dst_id'],
            self.node_features_df['node_id']
        ]).unique()
        self.node_encoder.fit(all_nodes)
        
        # Edge type'ları encode et
        self.edge_type_encoder.fit(self.edge_type_features_df['edge_type'])
        
        # Node feature'larını normalize et
        node_feat_cols = [col for col in self.node_features_df.columns if col != 'node_id']
        self.node_features = torch.tensor(
            self.node_features_df[node_feat_cols].fillna(-1).values,
            dtype=torch.float
        )
        
        # Edge feature'larını normalize et
        edge_feat_cols = [col for col in self.edge_type_features_df.columns if col != 'edge_type']
        self.edge_type_features = torch.tensor(
            self.edge_type_features_df[edge_feat_cols].fillna(-1).values,
            dtype=torch.float
        )
        
        # Edge'leri encode et
        self.edges_df['src_id'] = self.node_encoder.transform(self.edges_df['src_id'])
        self.edges_df['dst_id'] = self.node_encoder.transform(self.edges_df['dst_id'])
        self.edges_df['edge_type'] = self.edge_type_encoder.transform(self.edges_df['edge_type'])
        
        # Timestamp'leri normalize et
        self.edges_df['timestamp'] = (self.edges_df['timestamp'] - self.edges_df['timestamp'].min()) / \
                                   (self.edges_df['timestamp'].max() - self.edges_df['timestamp'].min())
    
    def get_temporal_graph(
        self,
        time_window: Tuple[float, float]
    ) -> Data:
        """
        Belirli bir zaman penceresi için temporal graph oluşturur.
        """
        mask = (self.edges_df['timestamp'] >= time_window[0]) & \
               (self.edges_df['timestamp'] < time_window[1])
        
        edge_index = torch.tensor(
            self.edges_df[mask][['src_id', 'dst_id']].values.T,
            dtype=torch.long
        )
        
        edge_type = torch.tensor(
            self.edges_df[mask]['edge_type'].values,
            dtype=torch.long
        )
        
        edge_attr = self.edge_type_features[edge_type]
        
        return Data(
            x=self.node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type
        )
    
    def get_prediction_batch(
        self,
        batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tahmin için batch oluşturur.
        """
        # Input verilerini encode et
        src_ids = self.node_encoder.transform(self.input_df['src_id'])
        dst_ids = self.node_encoder.transform(self.input_df['dst_id'])
        edge_types = self.edge_type_encoder.transform(self.input_df['edge_type'])
        
        # Timestamp'leri normalize et ve numpy array'e dönüştür
        start_times = ((self.input_df['start_time'] - self.edges_df['timestamp'].min()) / \
                      (self.edges_df['timestamp'].max() - self.edges_df['timestamp'].min())).values
        end_times = ((self.input_df['end_time'] - self.edges_df['timestamp'].min()) / \
                    (self.edges_df['timestamp'].max() - self.edges_df['timestamp'].min())).values
        
        # Batch'leri oluştur
        indices = np.random.permutation(len(self.input_df))
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            yield (
                torch.tensor(src_ids[batch_indices], dtype=torch.long),
                torch.tensor(dst_ids[batch_indices], dtype=torch.long),
                torch.tensor(edge_types[batch_indices], dtype=torch.long),
                torch.tensor(start_times[batch_indices], dtype=torch.float),
                torch.tensor(end_times[batch_indices], dtype=torch.float)
            ) 