"""
FedSage 论文复现 —— NeighGen 缺失邻居生成器
对应论文 Section 4.1, 公式 (3)(4)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class NeighborGenerator(nn.Module):
    """
    NeighGen: 给定节点特征，生成 num_pred 个缺失邻居的特征。
    对应论文公式 (3): fGen 部分。
    """
    def __init__(self, feature_dim, hidden_dim=64, num_pred=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_pred = num_pred
        
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_pred * feature_dim),
        )
    
    def forward(self, x):
        """
        参数: x: [batch_size, feature_dim]
        返回: [batch_size, num_pred, feature_dim]
        """
        z = self.encoder(x)
        noise = torch.randn_like(z)
        z_noisy = z + noise * 0.1
        out = self.generator(z_noisy)
        return out.view(-1, self.num_pred, self.feature_dim)


def neighgen_loss(generated, target_neighbors, num_real):
    """
    NeighGen 的训练损失，对应论文公式 (4)。
    
    参数:
        generated:        [batch, num_pred, feat_dim] 生成的邻居特征
        target_neighbors: [batch, num_pred, feat_dim] 真实邻居特征
        num_real:         [batch] 每个节点的真实邻居数量
    """
    batch_size = generated.size(0)
    total_loss = 0.0
    count = 0
    
    for i in range(batch_size):
        n_real = int(num_real[i].item())
        if n_real == 0:
            continue
        
        gen_i = generated[i]           # [num_pred, feat_dim]
        real_i = target_neighbors[i]   # [num_pred, feat_dim]
        
        for p in range(gen_i.size(0)):
            dists = torch.sum((gen_i[p].unsqueeze(0) - real_i[:n_real]) ** 2, dim=1)
            total_loss += dists.min()
        
        count += gen_i.size(0)
    
    if count == 0:
        return torch.tensor(0.0, requires_grad=True, device=generated.device)
    
    return total_loss / count


def prepare_neighgen_batch(sub_data, node_indices, num_pred):
    """
    为 NeighGen 准备训练 batch。
    
    对于每个采样的节点，找到它在子图中的真实邻居作为训练目标。
    """
    device = sub_data.x.device
    edge_index = sub_data.edge_index
    
    # 构建邻接表
    adj = {}
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        if src not in adj:
            adj[src] = []
        adj[src].append(dst)
    
    node_features = []
    target_neighbors = []
    num_real_list = []
    
    for node_id in node_indices.tolist():
        node_features.append(sub_data.x[node_id])
        
        neighbors = adj.get(node_id, [])
        n_real = min(len(neighbors), num_pred)
        
        if n_real > 0:
            nb_indices = neighbors[:num_pred]
            nb_feats = sub_data.x[nb_indices]
            
            if nb_feats.size(0) < num_pred:
                padding = torch.zeros(
                    num_pred - nb_feats.size(0), sub_data.x.size(1),
                    device=device
                )
                nb_feats = torch.cat([nb_feats, padding], dim=0)
        else:
            nb_feats = torch.zeros(num_pred, sub_data.x.size(1), device=device)
        
        target_neighbors.append(nb_feats)
        num_real_list.append(n_real)
    
    node_features = torch.stack(node_features)
    target_neighbors = torch.stack(target_neighbors)
    num_real = torch.tensor(num_real_list, dtype=torch.float, device=device)
    
    return node_features, target_neighbors, num_real
