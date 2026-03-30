"""
FedSage 论文复现 —— 本地训练器
论文 Section 5.1: "Optimization is done with Adam with a learning rate of 0.001."
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from utils.metrics import evaluate_model

class LocalTrainer:
    def __init__(self, model, data, lr=0.001, weight_decay=5e-4, device="cpu"):
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = model.to(device)
        self.data = data.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train_multiple_epochs(self, num_epochs):
        losses = []
        for _ in range(num_epochs):
            losses.append(self.train_epoch())
        return losses
    
    def reload_model_params(self, state_dict):
        """加载全局参数并重置 Adam 状态。"""
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
    
    def evaluate(self, mask=None):
        if mask is None:
            mask = self.data.test_mask
        return evaluate_model(self.model, self.data, mask, self.device)
