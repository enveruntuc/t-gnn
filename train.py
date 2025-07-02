import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score
import time
from typing import Tuple, List

from models.temporal_graphsage import TemporalGraphSAGE, prepare_temporal_batch, negative_sampling
from data.data_loader import TemporalGraphDataLoader

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_auc = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, auc: float, model: nn.Module) -> bool:
        if self.best_auc is None:
            self.best_auc = auc
            self.best_model_state = model.state_dict().copy()
        elif auc < self.best_auc + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_auc = auc
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        return self.early_stop

def evaluate(
    model: nn.Module,
    data_loader: TemporalGraphDataLoader,
    device: torch.device,
    batch_size: int = 32,
    num_neg_samples: int = 5
) -> Tuple[float, float]:
    """
    Modeli değerlendirir ve AUC ile loss değerlerini döndürür.
    """
    model.eval()
    criterion = nn.BCELoss()
    all_preds = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for src_ids, dst_ids, edge_types, start_times, end_times in data_loader.get_prediction_batch(batch_size):
            batch_preds = []
            batch_labels = []
            batch_loss = 0
            
            for i in range(len(src_ids)):
                time_window = (start_times[i].item(), end_times[i].item())
                graph = data_loader.get_temporal_graph(time_window)
                
                graph = graph.to(device)
                src_id = src_ids[i].to(device)
                dst_id = dst_ids[i].to(device)
                edge_type = edge_types[i].to(device)
                
                # Pozitif örnek
                pos_pred = model(
                    graph.x,
                    graph.edge_index,
                    graph.edge_type,
                    src_id.unsqueeze(0),
                    dst_id.unsqueeze(0)
                )
                
                # Negatif örnekler
                neg_edges = negative_sampling(
                    graph.edge_index,
                    graph.x.size(0),
                    num_neg_samples
                ).to(device)
                
                neg_pred = model(
                    graph.x,
                    graph.edge_index,
                    graph.edge_type,
                    neg_edges[0],
                    neg_edges[1]
                )
                
                # Loss hesapla
                pos_loss = criterion(pos_pred, torch.ones_like(pos_pred))
                neg_loss = criterion(neg_pred, torch.zeros_like(neg_pred))
                loss = pos_loss + neg_loss
                batch_loss += loss.item()
                
                # Tahminleri ve etiketleri topla
                batch_preds.extend([pos_pred.item()] + neg_pred.tolist())
                batch_labels.extend([1] + [0] * num_neg_samples)
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
            total_loss += batch_loss
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    auc = roc_auc_score(all_labels, all_preds)
    
    return auc, avg_loss

def train(
    model: nn.Module,
    data_loader: TemporalGraphDataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    batch_size: int = 32,
    num_neg_samples: int = 5,
    patience: int = 5,
    min_delta: float = 0.001
) -> Tuple[List[float], List[float]]:
    """
    Modeli AUC metriğine göre eğitir ve early stopping uygular.
    """
    model.train()
    criterion = nn.BCELoss()
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    
    train_aucs = []
    train_losses = []
    best_auc = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []
        
        # Eğitim
        for src_ids, dst_ids, edge_types, start_times, end_times in tqdm(
            data_loader.get_prediction_batch(batch_size),
            desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            optimizer.zero_grad()
            batch_preds = []
            batch_labels = []
            
            for i in range(len(src_ids)):
                time_window = (start_times[i].item(), end_times[i].item())
                graph = data_loader.get_temporal_graph(time_window)
                
                graph = graph.to(device)
                src_id = src_ids[i].to(device)
                dst_id = dst_ids[i].to(device)
                edge_type = edge_types[i].to(device)
                
                # Pozitif örnek
                pos_pred = model(
                    graph.x,
                    graph.edge_index,
                    graph.edge_type,
                    src_id.unsqueeze(0),
                    dst_id.unsqueeze(0)
                )
                
                # Negatif örnekler
                neg_edges = negative_sampling(
                    graph.edge_index,
                    graph.x.size(0),
                    num_neg_samples
                ).to(device)
                
                neg_pred = model(
                    graph.x,
                    graph.edge_index,
                    graph.edge_type,
                    neg_edges[0],
                    neg_edges[1]
                )
                
                # Loss hesapla
                pos_loss = criterion(pos_pred, torch.ones_like(pos_pred))
                neg_loss = criterion(neg_pred, torch.zeros_like(neg_pred))
                loss = pos_loss + neg_loss
                
                loss.backward()
                optimizer.step()
                
                # Tahminleri ve etiketleri topla
                batch_preds.extend([pos_pred.item()] + neg_pred.tolist())
                batch_labels.extend([1] + [0] * num_neg_samples)
                
                total_loss += loss.item()
                num_batches += 1
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
        
        # Epoch sonu metrikleri
        avg_loss = total_loss / num_batches
        train_auc = roc_auc_score(all_labels, all_preds)
        train_aucs.append(train_auc)
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train AUC: {train_auc:.4f}, Train Loss: {avg_loss:.4f}")
        
        # Early stopping kontrolü
        if early_stopping(train_auc, model):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            model.load_state_dict(early_stopping.best_model_state)
            break
    
    return train_aucs, train_losses

def predict(
    model: nn.Module,
    data_loader: TemporalGraphDataLoader,
    device: torch.device,
    batch_size: int = 32
) -> np.ndarray:
    """
    Test seti üzerinde tahmin yapar.
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for src_ids, dst_ids, edge_types, start_times, end_times in tqdm(
            data_loader.get_prediction_batch(batch_size),
            desc="Predicting"
        ):
            batch_preds = []
            
            for i in range(len(src_ids)):
                time_window = (start_times[i].item(), end_times[i].item())
                graph = data_loader.get_temporal_graph(time_window)
                
                graph = graph.to(device)
                src_id = src_ids[i].to(device)
                dst_id = dst_ids[i].to(device)
                edge_type = edge_types[i].to(device)
                
                pred = model(
                    graph.x,
                    graph.edge_index,
                    graph.edge_type,
                    src_id.unsqueeze(0),
                    dst_id.unsqueeze(0)
                )
                
                batch_preds.append(pred.item())
            
            predictions.extend(batch_preds)
    
    return np.array(predictions)

def main():
    # Cihaz seçimi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Veri yükleyici
    data_loader = TemporalGraphDataLoader(
        edges_path='edges_train_A.csv',
        node_features_path='node_features.csv',
        edge_type_features_path='edge_type_features.csv',
        input_path='input_A.csv'
    )
    
    # Model oluştur - daha küçük boyutlu
    model = TemporalGraphSAGE(
        node_feat_dim=data_loader.node_features.size(1),
        edge_type_feat_dim=data_loader.edge_type_features.size(1),
        hidden_dim=128,  # 256'dan 128'e düşürüldü
        num_layers=2,    # aynı kaldı
        dropout=0.1,     # aynı kaldı
        num_edge_types=248  # Dataset A'daki gerçek edge type sayısı
    ).to(device)
    
    # Optimizer - learning rate artırıldı
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Eğitim - daha az epoch ve daha büyük batch size
    print("Starting training...")
    train_aucs, train_losses = train(
        model,
        data_loader,
        optimizer,
        device,
        num_epochs=20,     # 100'den 20'ye düşürüldü
        batch_size=128,    # 32'den 128'e çıkarıldı
        num_neg_samples=3, # 5'ten 3'e düşürüldü
        patience=3,        # 5'ten 3'e düşürüldü
        min_delta=0.001    # aynı kaldı
    )
    
    # Eğitim sonuçlarını kaydet
    pd.DataFrame({
        'epoch': range(1, len(train_aucs) + 1),
        'auc': train_aucs,
        'loss': train_losses
    }).to_csv('training_history.csv', index=False)
    print("Training history saved to training_history.csv")
    
    # Tahmin
    print("Making predictions...")
    predictions = predict(model, data_loader, device, batch_size=128)  # batch size burada da artırıldı
    
    # Sonuçları kaydet
    pd.DataFrame({
        'probability': predictions
    }).to_csv('output_A.csv', index=False)
    print("Predictions saved to output_A.csv")

if __name__ == "__main__":
    main() 