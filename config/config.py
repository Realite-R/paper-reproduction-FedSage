"""
FedSage 论文复现 —— 全局配置文件
"""
import os
import torch

class Config:
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_path, "data")
    result_path = os.path.join(root_path, "results")
    log_path = os.path.join(result_path, "logs")
    checkpoint_path = os.path.join(result_path, "checkpoints")

    # 数据集: "cora", "citeseer", "pubmed", "amazon-computers"
    dataset = "amazon-computers"
    
    # 联邦学习
    num_owners = 3
    num_rounds = 100
    local_epochs = 3

    # GraphSage
    hidden_dim = 64
    num_layers = 2
    dropout = 0.5
    lr_classifier = 0.001  # 论文 Section 5.1: Adam with lr=0.001
    weight_decay = 5e-4

    # NeighGen
    num_pred = 5
    gen_hidden_dim = 64
    lr_generator = 0.001

    # 通用
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    log_interval = 10
    save_best_model = True
    num_runs = 5

    @classmethod
    def ensure_dirs(cls):
        for path in [cls.data_path, cls.log_path, cls.checkpoint_path]:
            os.makedirs(path, exist_ok=True)
    
    @classmethod
    def summary(cls):
        print("=" * 60)
        print("FedSage 实验配置摘要")
        print("=" * 60)
        print(f"  数据集:           {cls.dataset}")
        print(f"  客户端数量 M:     {cls.num_owners}")
        print(f"  联邦轮次:         {cls.num_rounds}")
        print(f"  本地 epoch 数:    {cls.local_epochs}")
        print(f"  隐藏维度:         {cls.hidden_dim}")
        print(f"  SAGEConv 层数:    {cls.num_layers}")
        print(f"  分类器学习率:     {cls.lr_classifier}")
        print(f"  NeighGen 预测数:  {cls.num_pred}")
        print(f"  随机种子:         {cls.seed}")
        print(f"  设备:             {cls.device}")
        print(f"  根目录:           {cls.root_path}")
        print("=" * 60)

cfg = Config()
