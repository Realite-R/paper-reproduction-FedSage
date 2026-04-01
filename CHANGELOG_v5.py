"""
FedSage v4 → v5 修改日志
========================

本文档记录了 v5 为对齐原仓库所做的所有修改，
每条修改对应之前发现的 7 个差异点。

=====================================================================
问题 1 [最高优先级]: FedSage+ 核心语义变了
=====================================================================
涉及文件: models/neighgen.py, trainers/fed_trainer.py

v4（错误）:
  - prepare_neighgen_batch 取子图内现有邻居作为训练目标
  - NeighborGenerator 只有 fGen（特征生成），没有 dGen（度数预测）
  - 损失只有一项：邻居特征 MSE
  - _augment_subgraph 给每个节点固定生成 num_pred 个邻居

v5（对齐原仓库）:
  - prepare_neighgen_batch 随机隐藏 h% 的节点，找"完整子图有但受损子图没有"
    的邻居作为训练目标（模拟跨子图缺失）
  - NeighborGenerator 包含 dGen + fGen + 可选分类头三个子模块
  - neighgen_loss 包含三部分：度数损失 + 特征损失 + 分类损失
  - _augment_subgraph 按 dGen 预测的缺失数量生成（不是固定 num_pred）

=====================================================================
问题 2 [高优先级]: 训练/评估 protocol 不同
=====================================================================
涉及文件: utils/graph_utils.py (split_masks), main.py

v4: split_masks 使用纯随机 np.random.permutation
v5: split_masks 改用 sklearn.train_test_split(stratify=labels)
    确保每个类在训练/验证/测试集中的比例大致相同

注意：我们保留了"全局统一划分→子图继承"的方案（避免 Bug 3 数据泄露），
但将随机划分升级为分层采样。

=====================================================================
问题 3 [高优先级]: 分类器 backbone 变了
=====================================================================
涉及文件: (暂未修改 sage.py)

原仓库用 StellarGraph 的 sampled GraphSAGE (num_samples=[5,5])，
我们仍用 PyG 的全图 SAGEConv。config 中已预留 num_samples=[5,5]
和 batch_size=64 参数，后续可改用 NeighborLoader 采样训练。

当前状态：config 已对齐参数，但训练方式尚未改为采样训练。
如果需要进一步对齐，可以在 local_trainer.py 中引入
torch_geometric.loader.NeighborLoader。

=====================================================================
问题 4 [高优先级]: Louvain 划分和类分布控制
=====================================================================
涉及文件: utils/graph_utils.py

v4: _adjust_communities 只做"合并最小/分裂最大"
v5: _adjust_and_balance 增加了三步：
    Phase 1: 合并/分裂到目标数量（同 v4）
    Phase 2: 平衡节点数量（从最大社区移节点到最小社区，直到差距 < 2*delta）
    Phase 3: 类覆盖修复（确保每个 owner 每个类至少有 min_samples_per_class=2
             个样本，不够时从富余的 owner 借）

=====================================================================
问题 5 [中优先级]: FedAvg 权重变了
=====================================================================
涉及文件: utils/fed_utils.py, trainers/fed_trainer.py, config/config.py

v4: fedavg_aggregate 按节点数加权（proportional）
v5: fedavg_aggregate 增加 mode 参数
    config.fedavg_weight = "equal"（1/M 等权，原仓库默认）
    fed_trainer 调用时传入 mode=self.cfg.fedavg_weight

=====================================================================
问题 6 [中优先级]: 默认超参没对齐
=====================================================================
涉及文件: config/config.py

v4 → v5 修改：
  seed:          42    → 2021
  local_epochs:  3     → 1
  num_rounds:    100   → 20
  weight_decay:  5e-4  → 1e-4
  新增 gen_rounds:      20（原仓库 gen_epochs=20）
  新增 hide_portion:    0.5（原仓库隐藏节点比例）

=====================================================================
问题 7 [低优先级]: Louvain random_state
=====================================================================
涉及文件: utils/graph_utils.py

v4: community_louvain.best_partition(G, random_state=seed)
v5: community_louvain.best_partition(G)  # 不传 random_state，匹配原仓库
"""
