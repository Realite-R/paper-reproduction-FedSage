"""
FedSage 论文复现 —— 图划分工具 (修复版)
=========================================

关键修复: 训练/测试集一致性
  之前的 bug: 全局图和子图各自独立做 60/20/20 划分，
  导致全局测试节点在子图中可能是训练节点 → 数据泄露。
  
  修复: 先在全局图上做一次划分，然后每个子图继承全局的划分。
  节点在全局是测试集，在子图里也必须是测试集。

使用方式:
    from utils.graph_utils import partition_graph, split_masks
    
    # 先在全局图上划分
    train_m, val_m, test_m = split_masks(data.x.size(0), 0.6, 0.2, seed=42)
    data.train_mask = train_m
    data.val_mask = val_m
    data.test_mask = test_m
    
    # 划分子图时自动继承全局 mask
    subgraphs = partition_graph(data, num_owners=3, seed=42)
    # subgraphs[i].train_mask 是从全局 mask 继承的，不是重新随机的
"""

import torch
import numpy as np
import networkx as nx
import community as community_louvain
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, subgraph


def partition_graph(data: Data, num_owners: int, seed: int = 42) -> list:
    """
    使用 Louvain 社区检测将全局图划分为 num_owners 个子图。
    
    关键: 如果 data 上已经有 train_mask/val_mask/test_mask，
    每个子图会继承对应节点的 mask（而不是重新随机划分）。
    
    参数:
        data:       PyG Data 对象（全局图，应已设置 train/val/test_mask）
        num_owners: 划分的子图数量 M
        seed:       随机种子
    
    返回:
        subgraphs: 长度为 M 的列表，每个元素是 PyG Data 对象
    """
    num_nodes = data.x.size(0)
    
    # ================================================================
    # Step 1: PyG 图 → NetworkX 图
    # ================================================================
    edge_index = to_undirected(data.edge_index)
    
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().numpy()
    G.add_edges_from(edges.tolist())
    
    # ================================================================
    # Step 2: 运行 Louvain 社区检测
    # ================================================================
    partition = community_louvain.best_partition(G, random_state=seed)
    
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)
    
    num_communities = len(communities)
    print(f"  Louvain 检测到 {num_communities} 个社区，目标 M = {num_owners}")
    
    # ================================================================
    # Step 3: 调整社区数量为 M
    # ================================================================
    community_list = _adjust_communities(communities, num_owners, seed)
    
    for i, comm_nodes in enumerate(community_list):
        print(f"  子图 {i}: {len(comm_nodes)} 个节点")
    
    # ================================================================
    # Step 4: 构建子图，继承全局 mask
    # ================================================================
    
    # 检查全局图是否已有 mask
    has_global_masks = (
        hasattr(data, "train_mask") and data.train_mask is not None
    )
    
    subgraphs = []
    
    for owner_id in range(num_owners):
        global_ids = sorted(community_list[owner_id])
        global_ids_tensor = torch.tensor(global_ids, dtype=torch.long)
        
        # 提取子图的边
        sub_edge_index, _, edge_mask = subgraph(
            subset=global_ids_tensor,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
            return_edge_mask=True,
        )
        
        # 提取子图的节点特征和标签
        sub_x = data.x[global_ids_tensor]
        sub_y = data.y[global_ids_tensor]
        
        sub_data = Data(
            x=sub_x,
            edge_index=sub_edge_index,
            y=sub_y,
        )
        
        # 保存全局编号映射
        sub_data.global_node_ids = global_ids_tensor
        sub_data.num_classes = data.num_classes
        
        # ★ 关键修复: 从全局 mask 继承，而不是重新随机划分
        if has_global_masks:
            sub_data.train_mask = data.train_mask[global_ids_tensor]
            sub_data.val_mask = data.val_mask[global_ids_tensor]
            sub_data.test_mask = data.test_mask[global_ids_tensor]
        
        subgraphs.append(sub_data)
    
    # ================================================================
    # Step 5: 统计缺失边和 mask 分布
    # ================================================================
    _print_missing_edge_stats(data, subgraphs)
    
    if has_global_masks:
        print(f"\n  数据划分 (继承自全局图):")
        for i, sg in enumerate(subgraphs):
            print(
                f"    子图 {i}: "
                f"训练 {sg.train_mask.sum().item()} | "
                f"验证 {sg.val_mask.sum().item()} | "
                f"测试 {sg.test_mask.sum().item()} | "
                f"总 {sg.x.size(0)}"
            )
    
    return subgraphs


def _adjust_communities(
    communities: dict,
    target_num: int,
    seed: int,
) -> list:
    """调整 Louvain 检测到的社区数量，使其恰好为 target_num。"""
    rng = np.random.RandomState(seed)
    comm_list = list(communities.values())
    
    # 社区太多 → 合并
    while len(comm_list) > target_num:
        comm_list.sort(key=len)
        smallest = comm_list.pop(0)
        comm_list[0].extend(smallest)
    
    # 社区太少 → 分裂
    while len(comm_list) < target_num:
        comm_list.sort(key=len, reverse=True)
        largest = comm_list.pop(0)
        rng.shuffle(largest)
        mid = len(largest) // 2
        comm_list.append(largest[:mid])
        comm_list.append(largest[mid:])
    
    return comm_list


def _print_missing_edge_stats(global_data: Data, subgraphs: list):
    """打印划分后缺失的跨子图边的统计信息。"""
    total_global_edges = global_data.edge_index.size(1)
    total_local_edges = sum(sg.edge_index.size(1) for sg in subgraphs)
    missing_edges = total_global_edges - total_local_edges
    missing_ratio = missing_edges / total_global_edges * 100
    
    print(f"\n  边统计:")
    print(f"    全局边总数:     {total_global_edges}")
    print(f"    保留的局部边:   {total_local_edges}")
    print(f"    缺失的跨子图边: {missing_edges} ({missing_ratio:.1f}%)")


def get_missing_neighbors(global_data: Data, sub_data: Data) -> dict:
    """计算某个子图中每个节点缺失了哪些邻居。"""
    global_ids = sub_data.global_node_ids.numpy()
    global_id_set = set(global_ids.tolist())
    
    global_edge_index = global_data.edge_index.numpy()
    global_adj = {}
    for i in range(global_edge_index.shape[1]):
        src, dst = global_edge_index[0, i], global_edge_index[1, i]
        if src not in global_adj:
            global_adj[src] = set()
        global_adj[src].add(dst)
    
    missing = {}
    for local_id, global_id in enumerate(global_ids):
        all_neighbors = global_adj.get(global_id, set())
        kept_neighbors = all_neighbors.intersection(global_id_set)
        num_missing = len(all_neighbors) - len(kept_neighbors)
        missing[local_id] = num_missing
    
    return missing


def split_masks(
    num_nodes: int,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> tuple:
    """
    为节点生成训练/验证/测试的 mask。
    
    应在全局图上调用一次，然后通过 partition_graph 继承到子图。
    不应对子图单独调用（那样会导致数据泄露）。
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(num_nodes)
    
    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train + num_val]] = True
    test_mask[indices[num_train + num_val:]] = True
    
    return train_mask, val_mask, test_mask
