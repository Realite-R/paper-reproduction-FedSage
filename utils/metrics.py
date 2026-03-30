"""FedSage 论文复现 —— 评估指标"""
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data

@torch.no_grad()
def evaluate_model(model, data, mask, device="cpu"):
    model.eval()
    model.to(device)
    data = data.to(device)
    mask = mask.to(device)
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[mask], data.y[mask]).item()
    pred = out[mask].argmax(dim=1).cpu()
    true = data.y[mask].cpu()
    acc = accuracy_score(true.numpy(), pred.numpy())
    f1 = f1_score(true.numpy(), pred.numpy(), average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1, "loss": loss}

def summarize_results(results_list):
    import numpy as np
    summary = {}
    keys = results_list[0].keys()
    for key in keys:
        values = [r[key] for r in results_list]
        summary[f"{key}_mean"] = np.mean(values)
        summary[f"{key}_std"] = np.std(values)
    return summary

def print_round_result(round_num, train_result, val_result, test_result=None):
    msg = (f"  Round {round_num:3d} | "
           f"Train Acc: {train_result['accuracy']:.4f} | "
           f"Val Acc: {val_result['accuracy']:.4f}")
    if test_result:
        msg += f" | Test Acc: {test_result['accuracy']:.4f}"
    print(msg)
