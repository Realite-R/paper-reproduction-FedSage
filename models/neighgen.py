"""
FedSage+ 论文复现 —— NeighGen 缺失邻居生成器 (v5 - 对齐原仓库语义)
====================================================================

原仓库的 NeighGen 训练流程（与我们 v4 的关键区别）：

  v4（错误语义）:
    - 直接取子图内现有邻居作为训练目标
    - 给每个节点固定生成 num_pred 个邻居
    - 损失函数只有特征 MSE 一项

  v5（正确语义，对齐原仓库）:
    1. 在子图上随机隐藏 h% 的节点，得到"受损子图"
    2. 对受损子图中的节点，找它在完整子图中有、但在受损子图中丢失的邻居
    3. 用这些"真实缺失邻居"作为生成器的训练目标
    4. NeighGen 包含两个子模块：
       - dGen: 预测缺失邻居的数量
       - fGen: 生成缺失邻居的特征
    5. 损失 = 缺失度数损失 + 邻居特征损失 + 节点类别损失
    6. 补图时，按 dGen 预测的缺失数量（而非固定 num_pred）生成邻居

对应论文 Section 4.1, 公式 (3)(4)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data


class NeighborGenerator(nn.Module):
    """
    NeighGen: 包含 dGen（度数预测）和 fGen（特征生成）两个子模块。

    dGen: 给定节点特征 → 预测该节点缺失了多少个邻居（回归值）
    fGen: 给定节点特征 → 生成 num_pred 个缺失邻居的特征向量
    """

    def __init__(self, feature_dim, hidden_dim=64, num_pred=5, num_classes=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_pred = num_pred
        self.num_classes = num_classes

        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # dGen: 预测缺失邻居数量（回归）
        self.degree_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU(),  # 数量不能为负
        )

        # fGen: 生成 num_pred 个邻居特征
        self.feature_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_pred * feature_dim),
        )

        # 可选：节点分类头（用于辅助分类损失）
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes),
            )
        else:
            self.classifier = None

    def forward(self, x):
        """
        参数: x: [batch_size, feature_dim]
        返回:
            pred_degree:  [batch_size, 1]  预测的缺失邻居数量
            gen_features: [batch_size, num_pred, feature_dim]  生成的邻居特征
            class_logits: [batch_size, num_classes] 或 None
        """
        z = self.encoder(x)

        # 加一点噪声增加生成多样性
        noise = torch.randn_like(z) * 0.1
        z_noisy = z + noise

        pred_degree = self.degree_predictor(z)
        gen_features = self.feature_generator(z_noisy)
        gen_features = gen_features.view(-1, self.num_pred, self.feature_dim)

        class_logits = None
        if self.classifier is not None:
            class_logits = self.classifier(z)

        return pred_degree, gen_features, class_logits


def neighgen_loss(
    pred_degree,
    gen_features,
    class_logits,
    target_degree,
    target_neighbors,
    target_labels,
    alpha_degree=1.0,
    alpha_feature=1.0,
    alpha_class=1.0,
):
    """
    NeighGen 的三部分联合损失。

    对应论文公式 (4):
      L = alpha_degree * L_degree + alpha_feature * L_feature + alpha_class * L_class

    参数:
        pred_degree:      [batch, 1]                 预测的缺失邻居数
        gen_features:     [batch, num_pred, feat_dim] 生成的邻居特征
        class_logits:     [batch, num_classes] 或 None
        target_degree:    [batch]                     真实缺失邻居数
        target_neighbors: [batch, num_pred, feat_dim] 真实缺失邻居特征
        target_labels:    [batch]                     节点类别标签
    """
    batch_size = gen_features.size(0)
    device = gen_features.device

    # ---- 损失 1: 缺失度数损失 (dGen) ----
    loss_degree = F.mse_loss(
        pred_degree.squeeze(-1),
        target_degree.float(),
    )

    # ---- 损失 2: 邻居特征损失 (fGen) ----
    # 对每个生成的邻居，找最接近的真实邻居计算 MSE
    total_feat_loss = torch.tensor(0.0, device=device)
    count = 0

    for i in range(batch_size):
        n_real = int(target_degree[i].item())
        n_real = min(n_real, target_neighbors.size(1))  # 安全截断
        if n_real == 0:
            continue

        gen_i = gen_features[i]           # [num_pred, feat_dim]
        real_i = target_neighbors[i, :n_real]  # [n_real, feat_dim]

        # 每个生成邻居找最近的真实邻居
        for p in range(gen_i.size(0)):
            dists = torch.sum((gen_i[p].unsqueeze(0) - real_i) ** 2, dim=1)
            total_feat_loss = total_feat_loss + dists.min()

        count += gen_i.size(0)

    if count > 0:
        loss_feature = total_feat_loss / count
    else:
        loss_feature = torch.tensor(0.0, device=device, requires_grad=True)

    # ---- 损失 3: 辅助分类损失 ----
    if class_logits is not None and target_labels is not None:
        # 只对有有效标签的节点计算
        valid_mask = target_labels >= 0
        if valid_mask.sum() > 0:
            loss_class = F.cross_entropy(
                class_logits[valid_mask],
                target_labels[valid_mask],
            )
        else:
            loss_class = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        loss_class = torch.tensor(0.0, device=device, requires_grad=True)

    total_loss = (
        alpha_degree * loss_degree
        + alpha_feature * loss_feature
        + alpha_class * loss_class
    )

    return total_loss, {
        "degree": loss_degree.item(),
        "feature": loss_feature.item(),
        "class": loss_class.item(),
        "total": total_loss.item(),
    }


def prepare_neighgen_batch(sub_data: Data, hide_portion: float, num_pred: int, rng=None):
    """
    为 NeighGen 准备训练 batch（对齐原仓库语义）。

    核心流程：
      1. 随机隐藏 hide_portion 比例的节点，模拟"跨子图丢失"
      2. 对保留的节点，计算它在完整子图中有、但在受损子图中丢失的邻居
      3. 返回：节点特征、缺失度数、缺失邻居特征、节点标签

    参数:
        sub_data:      子图的 PyG Data 对象
        hide_portion:  隐藏节点比例（0~1）
        num_pred:      最大预测缺失邻居数
        rng:           随机数生成器
    """
    if rng is None:
        rng = np.random.RandomState()

    device = sub_data.x.device
    num_nodes = sub_data.x.size(0)
    edge_index = sub_data.edge_index

    # ---- Step 1: 随机隐藏 h% 的节点 ----
    num_hide = max(1, int(num_nodes * hide_portion))
    all_indices = np.arange(num_nodes)
    rng.shuffle(all_indices)

    hidden_set = set(all_indices[:num_hide].tolist())
    visible_set = set(all_indices[num_hide:].tolist())

    # ---- Step 2: 构建完整子图的邻接表 ----
    adj = {}
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        if src not in adj:
            adj[src] = set()
        adj[src].add(dst)

    # ---- Step 3: 对可见节点，找缺失邻居 ----
    node_features = []
    target_degrees = []
    target_neighbors_list = []
    target_labels = []

    for node_id in sorted(visible_set):
        all_neighbors = adj.get(node_id, set())
        # 缺失邻居 = 完整子图中有、但被隐藏了的邻居
        missing_neighbors = all_neighbors.intersection(hidden_set)
        num_missing = len(missing_neighbors)

        node_features.append(sub_data.x[node_id])
        target_degrees.append(min(num_missing, num_pred))

        # 收集缺失邻居的真实特征作为训练目标
        if num_missing > 0:
            missing_list = sorted(missing_neighbors)[:num_pred]
            nb_feats = sub_data.x[missing_list]

            # padding 到 num_pred 长度
            if nb_feats.size(0) < num_pred:
                padding = torch.zeros(
                    num_pred - nb_feats.size(0), sub_data.x.size(1),
                    device=device,
                )
                nb_feats = torch.cat([nb_feats, padding], dim=0)
        else:
            nb_feats = torch.zeros(num_pred, sub_data.x.size(1), device=device)

        target_neighbors_list.append(nb_feats)
        target_labels.append(sub_data.y[node_id].item())

    node_features = torch.stack(node_features)
    target_degrees = torch.tensor(target_degrees, dtype=torch.float, device=device)
    target_neighbors = torch.stack(target_neighbors_list)
    target_labels = torch.tensor(target_labels, dtype=torch.long, device=device)

    return node_features, target_degrees, target_neighbors, target_labels
