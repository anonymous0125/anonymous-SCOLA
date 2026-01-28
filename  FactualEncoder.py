import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
from torch.optim import AdamW
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, Dataset, TensorDataset
from IPython.display import clear_output
import torch.nn.functional as F  
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics.pairwise import euclidean_distances
import random

# -----------------------------
# 1. Transformer Definition
# -----------------------------
class TSTransformerWithMask(nn.Module):
    def __init__(self, input_dim=9, max_time_steps=50, d_model=128, nhead=4, num_layers=3, embedding_size=3):
        super(TSTransformerWithMask, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embeddings = nn.Embedding(max_time_steps, d_model)
        
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, 24)
        self.norm1 = nn.LayerNorm(24)
        self.fc2 = nn.Linear(24, embedding_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, mask):
        """
        x: [batch_size, time_steps, input_dim]
        mask: [batch_size, time_steps] where 1 = real, 0 = padding
        """
        batch_size, seq_length, _ = x.shape

        x = self.input_projection(x)  # [B, T, D]
        position_ids = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings

        # Transformer expects key_padding_mask: [B, T] where True = mask out
        key_padding_mask = ~mask.bool()
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        # Apply masked average pooling
        mask = mask.unsqueeze(-1)  # [B, T, 1]
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # avoid divide by zero

        x = self.fc1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# -----------------------------
# 2. Soft Weighted InfoNCE Loss
# -----------------------------
def soft_weighted_info_nce_loss_full(anchor_embedding, all_embeddings, weights, temperature=1, device="cuda"):
    """
    anchor_embedding: [1, D]
    all_embeddings: [num_samples, D] 
    weights: [num_samples] 
    """
    # Step 1: Similarity calculation (negative Euclidean distance as similarity)
    dists = torch.norm(anchor_embedding - all_embeddings, p=2, dim=1) ** 2  # [num_samples]
    sims = -dists / temperature 

    # Step 2:  
    denominator = torch.sum(torch.exp(sims))  
    # Step 3: 
    softmax_probs = torch.exp(sims) / denominator  # [num_samples]

    # Step 4: CrossEntropy Loss
    
    # weights = torch.tensor(weights, dtype=torch.float32, device=anchor_embedding.device)
    weights = weights.to(dtype=torch.float32, device=anchor_embedding.device)
    
    weights = 1.0 / weights
    weights = weights / weights.sum()  # Normalize
    loss = -torch.sum(weights * torch.log(softmax_probs))  

    return loss

# -----------------------------
# 3. Batching Function for Online Training
# -----------------------------
def generate_batches_allmask_online(x, mask, env_vectors, batch_size):
    num_samples = x.shape[0]
    indices = list(range(num_samples))
    random.shuffle(indices)
    anchors, anchors_mask, allset, allset_mask, all_weights = [], [], [], [], []

    for i in range(batch_size):
        anchor_idx = indices[i]
        other_indices = indices[:anchor_idx] + indices[anchor_idx+1:]
        anchor = x[anchor_idx]
        anchor_m = mask[anchor_idx]
        anchor_env = env_vectors[anchor_idx]
        other_x = x[other_indices]
        other_mask = mask[other_indices]
        other_env = env_vectors[other_indices]
        weights = euclidean_distances([anchor_env], other_env)[0]
        weights[weights == 0] = 0.1

        anchors.append(anchor)
        anchors_mask.append(anchor_m)
        allset.append(other_x)
        allset_mask.append(other_mask)
        all_weights.append(weights)

    return anchors, anchors_mask, allset, allset_mask, all_weights

# -----------------------------
# 4. Training Function
# -----------------------------
def train_epoch_efficient_online(model, x, mask, env_vectors, optimizer, batch_size=32, device="cuda"):
    model.train()
    total_loss = 0
    anchors, anchors_mask, allset, allset_mask, all_weights = generate_batches_allmask_online(x, mask, env_vectors, batch_size)

    time_steps = x.shape[1] #50
    input_dim = x.shape[2] #9

    for i in range(batch_size):
        anchor = torch.tensor(anchors[i], dtype=torch.float32).view(1, time_steps, input_dim).to(device)
        anchor_mask = torch.tensor(anchors_mask[i], dtype=torch.float32).view(1, time_steps).to(device)

        all_x = torch.tensor(allset[i], dtype=torch.float32).to(device)  # [99, 50, 9]
        all_m = torch.tensor(allset_mask[i], dtype=torch.float32).to(device)  # [99, 50]
        weight = torch.tensor(all_weights[i], dtype=torch.float32).to(device)  # [99]

        # anchor_embedding = model(anchor, anchor_mask)  # [1, 3]
        # all_embeddings = model(all_x, all_m)  # [99, 3]
        anchor_embedding = model(anchor.to(device), anchor_mask.to(device))
        all_embeddings = model(all_x.to(device), all_m.to(device))

        loss = soft_weighted_info_nce_loss_full(anchor_embedding, all_embeddings, weight.unsqueeze(0))  # [1, 99]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / batch_size
    # print(f"[Transformer] Epoch Train Loss: {avg_loss:.4f}")
    return avg_loss

# -----------------------------
# 5. Positive/Negative Batching for Validation
# -----------------------------
def generate_batches_pos_neg_online(x, mask, env_vectors, batch_size, threshold=0.6):
    num_samples = x.shape[0]
    indices = list(range(num_samples))
    random.shuffle(indices)

    anchors, anchors_mask = [], []
    pos_set, neg_set = [], []
    pos_mask_set, neg_mask_set = [], []

    distance_matrix = euclidean_distances(env_vectors)  # [N, N]

    for i in range(batch_size):
        anchor_idx = indices[i]
        anchor = x[anchor_idx]
        anchor_m = mask[anchor_idx]

        # Precise division of positive and negative samples (all retained)
        pos_indices = [j for j in range(num_samples) if j != anchor_idx and distance_matrix[anchor_idx, j] < threshold]
        neg_indices = [j for j in range(num_samples) if j != anchor_idx and distance_matrix[anchor_idx, j] >= threshold]

        # Convert to ndarray slices
        pos_set.append(x[pos_indices])            # shape: [?, 50, 9]
        pos_mask_set.append(mask[pos_indices])    # shape: [?, 50]
        neg_set.append(x[neg_indices])            # shape: [?, 50, 9]
        neg_mask_set.append(mask[neg_indices])    # shape: [?, 50]

        anchors.append(anchor)
        anchors_mask.append(anchor_m)

    return anchors, pos_set, neg_set, anchors_mask, pos_mask_set, neg_mask_set


# -----------------------------
# 6. Evaluation Function
# -----------------------------
def validate_epoch_efficient_online(model, x, mask, env_vectors, batch_size=32, device="cuda"):
    model.eval()
    total_loss, total_ap, total_an = 0.0, 0.0, 0.0

    anchors, positives, negatives, anchors_mask, pos_mask, neg_mask = \
        generate_batches_pos_neg_online(x, mask, env_vectors, batch_size)

    time_steps = x.shape[1]
    input_dim = x.shape[2]

    with torch.no_grad():
        for i in range(batch_size):
            anchor = torch.tensor(anchors[i], dtype=torch.float32).view(1, time_steps, input_dim).to(device)
            anchor_m = torch.tensor(anchors_mask[i], dtype=torch.float32).view(1, time_steps).to(device)

            pos_x = torch.tensor(positives[i], dtype=torch.float32).to(device)  # [num_pos, 50, 9]
            pos_m = torch.tensor(pos_mask[i], dtype=torch.float32).to(device)

            neg_x = torch.tensor(negatives[i], dtype=torch.float32).to(device)  # [num_neg, 50, 9]
            neg_m = torch.tensor(neg_mask[i], dtype=torch.float32).to(device)

            anchor_emb = model(anchor, anchor_m)  # [1, 3]
            pos_emb = model(pos_x, pos_m)  # [num_pos, 3]
            neg_emb = model(neg_x, neg_m)  # [num_neg, 3]

            # Compute average distances
            pos_dist = torch.norm(anchor_emb - pos_emb, p=2, dim=1).mean()
            neg_dist = torch.norm(anchor_emb - neg_emb, p=2, dim=1).mean()

            loss = pos_dist - neg_dist  # Want pos close, neg far
            total_loss += loss.item()
            total_ap += pos_dist.item()
            total_an += neg_dist.item()

    avg_loss = total_loss / batch_size
    avg_ap = total_ap / batch_size
    avg_an = total_an / batch_size

    return avg_loss, avg_ap, avg_an