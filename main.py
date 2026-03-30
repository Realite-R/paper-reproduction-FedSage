"""
FedSage 论文复现 —— 主入口脚本 (修复版)
=========================================

关键修复: 数据划分顺序
  之前: 先划分子图，再各自独立做 60/20/20 → 数据泄露
  现在: 先在全局图上做一次 60/20/20，再划分子图继承全局 mask

运行方式:
    cd fedsage_final/
    python main.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import cfg
from data.data_loader import load_dataset, print_dataset_info
from utils.seed import set_seed
from utils.graph_utils import partition_graph, split_masks, get_missing_neighbors


def main():
    """主函数: 串联整个实验 pipeline。"""
    
    print("\n" + "=" * 60)
    print("  FedSage 论文复现 (PyG 版本)")
    print("=" * 60)
    
    cfg.summary()
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    
    # ==================================================================
    # Step 1: 加载全局图数据集
    # ==================================================================
    print("\n[Step 1] 加载数据集...")
    
    data = load_dataset(cfg.dataset, cfg.data_path)
    print_dataset_info(data, cfg.dataset)
    
    num_features = data.x.size(1)
    num_classes = data.num_classes
    
    print(f"  模型输入维度: {num_features}")
    print(f"  模型输出维度: {num_classes}")
    
    # ==================================================================
    # Step 2: ★ 先在全局图上做一次训练/验证/测试划分
    # ==================================================================
    # 这是修复的关键!
    # 论文 Section 5.1: "training-validation-testing ratio is 60%-20%-20%"
    # 必须先做全局划分，再让子图继承，确保:
    #   - 全局测试节点在所有子图中也是测试节点
    #   - FedSage 训练时绝不会看到测试节点的标签
    #   - GlobSage 和 FedSage 在完全相同的测试集上评估
    
    print(f"\n[Step 2] 在全局图上创建统一的数据划分...")
    
    train_mask, val_mask, test_mask = split_masks(
        num_nodes=data.x.size(0),
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )
    
    # 将 mask 挂到全局图上
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    print(
        f"  全局图: "
        f"训练 {train_mask.sum().item()} | "
        f"验证 {val_mask.sum().item()} | "
        f"测试 {test_mask.sum().item()} | "
        f"总 {data.x.size(0)}"
    )
    
    # ==================================================================
    # Step 3: 用 Louvain 划分子图（自动继承全局 mask）
    # ==================================================================
    print(f"\n[Step 3] 将全局图划分为 {cfg.num_owners} 个子图...")
    
    # partition_graph 检测到 data 上已有 mask，
    # 会自动将每个子图的 mask 设为对应全局节点的 mask
    subgraphs = partition_graph(
        data=data,
        num_owners=cfg.num_owners,
        seed=cfg.seed,
    )
    
    # ==================================================================
    # Step 4: 分析缺失邻居
    # ==================================================================
    print(f"\n[Step 4] 分析各子图的缺失邻居情况...")
    
    for i, sg in enumerate(subgraphs):
        missing = get_missing_neighbors(data, sg)
        total_missing = sum(missing.values())
        nodes_with_missing = sum(1 for v in missing.values() if v > 0)
        avg_missing = total_missing / len(missing) if missing else 0
        
        print(
            f"  子图 {i}: "
            f"{nodes_with_missing}/{len(missing)} 个节点有缺失邻居, "
            f"平均每节点缺失 {avg_missing:.2f} 个邻居"
        )
    
    # ==================================================================
    # Step 5: 运行全部实验
    # ==================================================================
    print(f"\n[Step 5] 运行实验...")
    
    from utils.result_saver import ResultSaver
    from trainers.fed_trainer import FedTrainer
    
    saver = ResultSaver(cfg)
    
    trainer = FedTrainer(
        subgraphs=subgraphs,
        global_data=data,
        cfg=cfg,
        saver=saver,
    )
    
    all_results = trainer.run_all_experiments()
    
    return all_results


if __name__ == "__main__":
    results = main()
