"""
FedSage 论文复现 —— FedAvg 参数聚合 (v5)

修复：原仓库用 1/num_owners 等权平均，不是按节点数加权。
"""
import torch
from collections import OrderedDict


def fedavg_aggregate(client_models, mode="equal", client_weights=None):
    """
    FedAvg 参数聚合。

    参数:
        client_models: 客户端模型列表
        mode:  "equal" = 1/M 等权（原仓库默认）
               "proportional" = 按 client_weights 加权
        client_weights: 仅在 mode="proportional" 时使用
    """
    num_clients = len(client_models)

    if mode == "equal":
        weights = [1.0 / num_clients] * num_clients
    elif mode == "proportional":
        if client_weights is None:
            weights = [1.0 / num_clients] * num_clients
        else:
            total = sum(client_weights)
            weights = [w / total for w in client_weights]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    global_state = OrderedDict()
    first_state = client_models[0].state_dict()
    for key in first_state:
        global_state[key] = torch.zeros_like(first_state[key], dtype=torch.float32)

    for cid in range(num_clients):
        state = client_models[cid].state_dict()
        w = weights[cid]
        for key in global_state:
            global_state[key] += w * state[key].float()

    return global_state


def distribute_global_model(global_state, client_models):
    """把全局参数下发给所有客户端。"""
    for model in client_models:
        model.load_state_dict(global_state)
