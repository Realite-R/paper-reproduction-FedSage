"""FedSage 论文复现 —— FedAvg 参数聚合"""
import torch
from collections import OrderedDict

def fedavg_aggregate(client_models, client_weights=None):
    num_clients = len(client_models)
    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients
    else:
        total = sum(client_weights)
        client_weights = [w / total for w in client_weights]
    
    global_state = OrderedDict()
    first_state = client_models[0].state_dict()
    for key in first_state:
        global_state[key] = torch.zeros_like(first_state[key], dtype=torch.float32)
    
    for cid in range(num_clients):
        state = client_models[cid].state_dict()
        w = client_weights[cid]
        for key in global_state:
            global_state[key] += w * state[key].float()
    
    return global_state

def distribute_global_model(global_state, client_models):
    for model in client_models:
        model.load_state_dict(global_state)

def create_client_models(model_class, model_kwargs, num_clients):
    return [model_class(**model_kwargs) for _ in range(num_clients)]

def initialize_global_model(model_class, model_kwargs):
    return model_class(**model_kwargs)
