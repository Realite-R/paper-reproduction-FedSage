"""
FedSage 论文复现 —— 图划分工具 (v5 - 对齐原仓库)
===================================================

修复项：
  1. Louvain 不传 random_state（原仓库行为）
  2. 增加节点数量平衡（原仓库 owner_nodes_len ± delta）
  3. 增加类覆盖修复（每个 owner 每个类至少 min_samples_per_class 个样本）
  4. split_masks 改为 stratified 分层采样
"""
import torch
import numpy as np
import networkx as nx
import community as community_louvain
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, subgraph
from sklearn.model_selection import train_test_split


def partition_graph(
    data: Data,
    num_owners: int,
    min_samples_per_class: int = 2,
    balance: bool = True,
    seed: int = 2021,
) -> list:
    """
    使用 Louvain 社区检测将全局图划分为 num_owners 个子图。

    对齐原仓库：
      - Louvain 不传 random_state
      - 平衡各 owner 节点数量
      - 确保每个 owner 对每个类至少有 min_samples_per_class 个样本
      - 子图继承全局的 train/val/test mask
    """
    num_nodes = data.x.size(0)
    labels = data.y.numpy()

    # ================================================================
    # Step 1: PyG → NetworkX
    # ================================================================
    edge_index = to_undirected(data.edge_index)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().numpy()
    G.add_edges_from(edges.tolist())

    # ================================================================
    # Step 2: Louvain 社区检测（不传 random_state，匹配原仓库）
    # ================================================================
    partition = community_louvain.best_partition(G)

    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)

    num_communities = len(communities)
    print(f"  Louvain 检测到 {num_communities} 个社区，目标 M = {num_owners}")

    # ================================================================
    # Step 3: 调整社区数 → M，平衡节点数，修复类覆盖
    # ================================================================
    rng = np.random.RandomState(seed)
    community_list = _adjust_and_balance(
        communities, num_owners, num_nodes, labels,
        min_samples_per_class, balance, rng,
    )

    for i, comm_nodes in enumerate(community_list):
        print(f"  子图 {i}: {len(comm_nodes)} 个节点")

    # ================================================================
    # Step 4: 构建子图，继承全局 mask
    # ================================================================
    has_global_masks = (
        hasattr(data, "train_mask") and data.train_mask is not None
    )

    subgraphs = []

    for owner_id in range(num_owners):
        global_ids = sorted(community_list[owner_id])
        global_ids_tensor = torch.tensor(global_ids, dtype=torch.long)

        sub_edge_index, _, edge_mask = subgraph(
            subset=global_ids_tensor,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
            return_edge_mask=True,
        )

        sub_x = data.x[global_ids_tensor]
        sub_y = data.y[global_ids_tensor]

        sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)
        sub_data.global_node_ids = global_ids_tensor
        sub_data.num_classes = data.num_classes

        if has_global_masks:
            sub_data.train_mask = data.train_mask[global_ids_tensor]
            sub_data.val_mask = data.val_mask[global_ids_tensor]
            sub_data.test_mask = data.test_mask[global_ids_tensor]

        subgraphs.append(sub_data)

    # ================================================================
    # Step 5: 统计
    # ================================================================
    _print_missing_edge_stats(data, subgraphs)

    if has_global_masks:
        print(f"\n  数据划分 (继承自全局图):")
        for i, sg in enumerate(subgraphs):
            n_classes_present = len(torch.unique(sg.y))
            print(
                f"    子图 {i}: "
                f"训练 {sg.train_mask.sum().item()} | "
                f"验证 {sg.val_mask.sum().item()} | "
                f"测试 {sg.test_mask.sum().item()} | "
                f"总 {sg.x.size(0)} | "
                f"类别覆盖 {n_classes_present}/{data.num_classes}"
            )

    return subgraphs


def _adjust_and_balance(
    communities: dict,
    target_num: int,
    total_nodes: int,
    labels: np.ndarray,
    min_samples_per_class: int,
    balance: bool,
    rng: np.random.RandomState,
) -> list:
    """
    调整社区数量为 target_num，平衡节点数，修复类覆盖。

    对齐原仓库的逻辑：
      1. 先合并/分裂到目标数量
      2. 在 owner 之间移动节点使数量接近 total_nodes/M
      3. 确保每个 owner 对每个类至少有 min_samples_per_class 个样本
    """
    comm_list = list(communities.values())

    # --- Phase 1: 调整到目标数量 ---
    while len(comm_list) > target_num:
        comm_list.sort(key=len)
        smallest = comm_list.pop(0)
        comm_list[0].extend(smallest)

    while len(comm_list) < target_num:
        comm_list.sort(key=len, reverse=True)
        largest = comm_list.pop(0)
        rng.shuffle(largest)
        mid = len(largest) // 2
        comm_list.append(largest[:mid])
        comm_list.append(largest[mid:])

    # --- Phase 2: 平衡节点数量 ---
    if balance:
        target_size = total_nodes // target_num
        delta = max(target_size // 5, 10)  # 允许 ±20% 波动

        for _ in range(50):  # 最多迭代 50 次
            sizes = [len(c) for c in comm_list]
            max_idx = np.argmax(sizes)
            min_idx = np.argmin(sizes)

            if sizes[max_idx] - sizes[min_idx] <= 2 * delta:
                break

            # 从最大的社区移一个节点到最小的
            node = comm_list[max_idx].pop()
            comm_list[min_idx].append(node)

    # --- Phase 3: 类覆盖修复 ---
    num_classes = len(np.unique(labels))

    for owner_id in range(len(comm_list)):
        owner_nodes = comm_list[owner_id]
        owner_labels = labels[owner_nodes]

        for cls in range(num_classes):
            cls_count = np.sum(owner_labels == cls)

            if cls_count < min_samples_per_class:
                needed = min_samples_per_class - cls_count
                # 从其他 owner 中寻找可以借的节点
                for donor_id in range(len(comm_list)):
                    if donor_id == owner_id:
                        continue

                    donor_nodes = comm_list[donor_id]
                    donor_labels = labels[donor_nodes]
                    donor_cls_mask = (donor_labels == cls)
                    donor_cls_count = np.sum(donor_cls_mask)

                    # 只在 donor 有富余时才借
                    if donor_cls_count > min_samples_per_class:
                        can_give = min(needed, donor_cls_count - min_samples_per_class)
                        donor_cls_indices = np.where(donor_cls_mask)[0]
                        give_indices = donor_cls_indices[:can_give]

                        for idx in sorted(give_indices, reverse=True):
                            node = donor_nodes.pop(idx)
                            comm_list[owner_id].append(node)
                            needed -= 1

                        if needed <= 0:
                            break

                # 更新 owner 的标签缓存
                owner_labels = labels[comm_list[owner_id]]

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
    labels: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    seed: int = 2021,
) -> tuple:
    """
    为节点生成 stratified 训练/验证/测试 mask。

    修复：之前是纯随机 permutation，现在改为分层采样（stratified），
    确保每个类在训练/验证/测试集中的比例大致相同。
    """
    labels_np = labels.numpy()
    indices = np.arange(num_nodes)

    # 第一次分割：训练集 vs (验证+测试)
    train_idx, rest_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=labels_np,
        random_state=seed,
    )

    # 第二次分割：验证集 vs 测试集
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        rest_idx,
        train_size=val_ratio_adjusted,
        stratify=labels_np[rest_idx],
        random_state=seed,
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask
