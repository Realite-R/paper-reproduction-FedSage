"""
FedSage 论文复现 —— 全局配置文件 (v5 - 对齐原仓库)
===================================================

对齐项（与原仓库 main 分支对照）：
  - seed: 2021
  - local_epochs: 1          (原仓库 epochs_local=1)
  - num_rounds: 20           (原仓库 epoch_classifier=20)
  - gen_rounds: 20           (原仓库 gen_epochs=20)
  - weight_decay: 1e-4
  - num_samples: [5,5]       (原仓库 sampled GraphSAGE)
  - fedavg_weight: "equal"   (原仓库 1/num_owners)
  - hide_portion: 0.5        (原仓库隐藏节点比例)
"""
import os
import torch


class Config:
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_path, "data")
    result_path = os.path.join(root_path, "results")
    log_path = os.path.join(result_path, "logs")
    checkpoint_path = os.path.join(result_path, "checkpoints")

    # 数据集: "cora", "citeseer", "pubmed"
    dataset = "citeseer"

    # ===================== 联邦学习 =====================
    num_owners = 3
    num_rounds = 100               # 原仓库 epoch_classifier=20
    local_epochs = 3              # 原仓库 epochs_local=1
    fedavg_weight = "equal"       # "equal" = 1/M (原仓库); "proportional" = 按节点数加权

    # ===================== GraphSage =====================
    hidden_dim = 64
    num_layers = 2
    dropout = 0.5
    lr_classifier = 0.001
    weight_decay = 1e-4           # 原仓库默认（之前我们误用 5e-4）
    num_samples = [5, 5]          # 原仓库 sampled GraphSAGE 的采样邻居数
    batch_size = 64

    # ===================== NeighGen =====================
    num_pred = 5                  # 最大预测缺失邻居数
    gen_hidden_dim = 64
    lr_generator = 0.001
    gen_rounds = 50               # 原仓库 gen_epochs=20
    hide_portion = 0.5            # 隐藏节点比例 h（论文 Section 4.2）

    # ===================== 通用 =====================
    seed = 2021                   # 原仓库默认种子
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    log_interval = 5
    save_best_model = True
    num_runs = 5

    # ===================== Louvain 划分 =====================
    min_samples_per_class = 2     # 每个 owner 每个类至少有 2 个样本
    balance_partition = True      # 是否平衡各 owner 节点数量

    @classmethod
    def ensure_dirs(cls):
        for path in [cls.data_path, cls.log_path, cls.checkpoint_path]:
            os.makedirs(path, exist_ok=True)

    @classmethod
    def summary(cls):
        print("=" * 60)
        print("  FedSage 实验配置摘要 (v5 - 对齐原仓库)")
        print("=" * 60)
        print(f"  数据集:           {cls.dataset}")
        print(f"  客户端数量 M:     {cls.num_owners}")
        print(f"  联邦轮次:         {cls.num_rounds}")
        print(f"  本地 epoch 数:    {cls.local_epochs}")
        print(f"  NeighGen 轮次:    {cls.gen_rounds}")
        print(f"  隐藏维度:         {cls.hidden_dim}")
        print(f"  采样邻居数:       {cls.num_samples}")
        print(f"  分类器学习率:     {cls.lr_classifier}")
        print(f"  weight_decay:     {cls.weight_decay}")
        print(f"  FedAvg 权重:      {cls.fedavg_weight}")
        print(f"  隐藏节点比例 h:   {cls.hide_portion}")
        print(f"  随机种子:         {cls.seed}")
        print(f"  设备:             {cls.device}")
        print("=" * 60)


cfg = Config()
