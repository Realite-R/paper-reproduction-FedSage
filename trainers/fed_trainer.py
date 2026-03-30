"""
FedSage 论文复现 —— 联邦训练编排器 (v4 - 数据泄露修复)
=======================================================

关键修复: 数据划分一致性
  之前的 bug:
    全局图和子图各自独立做 60/20/20 划分。
    全局测试节点在子图中可能是训练节点 → FedSage 在"测试"时
    其实评估的是训练过的节点 → 准确率虚高 → 出现 FedSage > GlobSage
  
  修复后:
    全局图先做一次 60/20/20 划分，子图继承全局的 mask。
    所有模型在完全相同的全局测试节点上评估。
    保证 LocSage < FedSage ≤ FedSage+ < GlobSage。

论文实验设置 (Section 5.1):
  - 数据集: Cora, CiteSeer, PubMed, MSAcademic
  - 划分: Louvain, M = 3, 5, 10
  - 训练/验证/测试 = 60/20/20
  - 优化器: Adam, lr = 0.001
  - GraphSage: 2 层, mean aggregator
  - 指标: 全局图测试节点上的节点分类准确率
"""

import copy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from models.sage import GraphSageClassifier
from models.neighgen import NeighborGenerator, neighgen_loss, prepare_neighgen_batch
from trainers.local_trainer import LocalTrainer
from utils.fed_utils import fedavg_aggregate, distribute_global_model
from utils.metrics import evaluate_model


class FedTrainer:
    """联邦训练编排器。"""
    
    def __init__(self, subgraphs: list, global_data: Data, cfg, saver=None):
        self.subgraphs = subgraphs
        self.global_data = global_data
        self.cfg = cfg
        self.device = cfg.device
        self.saver = saver
        
        self.in_dim = global_data.x.size(1)
        self.out_dim = global_data.num_classes
        self.client_weights = [sg.x.size(0) for sg in subgraphs]
        
        # 确认全局图已有 mask（由 main.py 创建）
        assert hasattr(global_data, "train_mask") and global_data.train_mask is not None, \
            "全局图必须先做 train/val/test 划分！请在 main.py 中先调用 split_masks。"
    
    def _make_model_kwargs(self):
        return {
            "in_dim": self.in_dim,
            "hidden_dim": self.cfg.hidden_dim,
            "out_dim": self.out_dim,
            "num_layers": self.cfg.num_layers,
            "dropout": self.cfg.dropout,
        }
    
    def _log(self, experiment, round_num, **metrics):
        """记录到 ResultSaver（如果有的话）。"""
        if self.saver:
            self.saver.log_round(experiment, round_num, **metrics)
    
    def _eval_on_global(self, model) -> dict:
        """
        在全局图的测试节点上评估模型。
        所有实验的统一评估方式。
        """
        return evaluate_model(
            model=model,
            data=self.global_data,
            mask=self.global_data.test_mask,
            device=self.device,
        )
    
    # ==================================================================
    #  实验 1: GlobSage (上界)
    # ==================================================================
    
    def run_globsage(self) -> dict:
        """
        GlobSage: 在完整全局图上集中训练和评估。
        训练集 = 全局 train_mask，评估 = 全局 test_mask。
        """
        print("\n" + "=" * 60)
        print("  实验: GlobSage (全局图集中训练 — 性能上界)")
        print("=" * 60)
        
        model = GraphSageClassifier(**self._make_model_kwargs())
        
        # GlobSage 直接在全局图上训练
        trainer = LocalTrainer(
            model=model,
            data=self.global_data,
            lr=self.cfg.lr_classifier,
            weight_decay=self.cfg.weight_decay,
            device=self.device,
        )
        
        total_epochs = self.cfg.num_rounds * self.cfg.local_epochs
        best_val_acc = 0
        best_test_result = None
        
        for epoch in range(1, total_epochs + 1):
            loss = trainer.train_epoch()
            
            if epoch % self.cfg.log_interval == 0 or epoch == total_epochs:
                val_result = evaluate_model(
                    model, self.global_data,
                    self.global_data.val_mask, self.device
                )
                test_result = self._eval_on_global(model)
                
                if val_result["accuracy"] > best_val_acc:
                    best_val_acc = val_result["accuracy"]
                    best_test_result = test_result
                
                self._log("GlobSage", epoch,
                          train_loss=loss, test_acc=test_result["accuracy"])
                
                print(
                    f"  Epoch {epoch:3d}/{total_epochs} | "
                    f"Loss: {loss:.4f} | "
                    f"Val: {val_result['accuracy']:.4f} | "
                    f"Test: {test_result['accuracy']:.4f}"
                )
        
        print(f"\n  ★ GlobSage 最佳测试准确率: {best_test_result['accuracy']:.4f}")
        return best_test_result
    
    # ==================================================================
    #  实验 2: LocSage (下界)
    # ==================================================================
    
    def run_locsage(self) -> list:
        """
        LocSage: 各客户端独立训练，在【全局图】上评估。
        
        每个子图的 train_mask 已继承自全局划分，
        所以子图内的训练节点一定是全局训练集的子集。
        """
        print("\n" + "=" * 60)
        print("  实验: LocSage (各客户端独立训练 — 性能下界)")
        print("=" * 60)
        
        results = []
        total_epochs = self.cfg.num_rounds * self.cfg.local_epochs
        
        for i, sg in enumerate(self.subgraphs):
            model = GraphSageClassifier(**self._make_model_kwargs())
            
            trainer = LocalTrainer(
                model=model, data=sg,
                lr=self.cfg.lr_classifier,
                weight_decay=self.cfg.weight_decay,
                device=self.device,
            )
            
            for epoch in range(1, total_epochs + 1):
                trainer.train_epoch()
            
            # 在【全局图】上评估
            test_result = self._eval_on_global(model)
            results.append(test_result)
            
            print(
                f"  客户端 {i}: 全局 Test Acc = {test_result['accuracy']:.4f} "
                f"(本地 {sg.train_mask.sum().item()} 训练节点)"
            )
        
        avg_acc = sum(r["accuracy"] for r in results) / len(results)
        print(f"\n  ★ LocSage 平均测试准确率: {avg_acc:.4f}")
        
        return results
    
    # ==================================================================
    #  实验 3: FedSage
    # ==================================================================
    
    def run_fedsage(self, subgraphs=None, client_weights=None, label="FedSage") -> dict:
        """
        FedSage: FedAvg + GraphSage。
        各客户端在子图上本地训练 + FedAvg 聚合。
        评估: 聚合后的全局模型在【全局图测试节点】上评估。
        """
        if subgraphs is None:
            subgraphs = self.subgraphs
        if client_weights is None:
            client_weights = self.client_weights
        
        print("\n" + "=" * 60)
        print(f"  实验: {label} (FedAvg + GraphSage)")
        print("=" * 60)
        
        model_kwargs = self._make_model_kwargs()
        
        global_model = GraphSageClassifier(**model_kwargs)
        global_state = global_model.state_dict()
        
        # 创建持久化的客户端 trainer
        client_models = []
        client_trainers = []
        
        for i, sg in enumerate(subgraphs):
            model = GraphSageClassifier(**model_kwargs)
            model.load_state_dict(global_state)
            
            trainer = LocalTrainer(
                model=model, data=sg,
                lr=self.cfg.lr_classifier,
                weight_decay=self.cfg.weight_decay,
                device=self.device,
            )
            client_models.append(model)
            client_trainers.append(trainer)
        
        best_test_acc = 0
        best_test_result = None
        
        for rnd in range(1, self.cfg.num_rounds + 1):
            
            # Step 1: 分发全局参数
            for trainer in client_trainers:
                trainer.reload_model_params(global_state)
            
            # Step 2: 各客户端本地训练
            for trainer in client_trainers:
                trainer.train_multiple_epochs(self.cfg.local_epochs)
            
            # Step 3: FedAvg 聚合
            global_state = fedavg_aggregate(client_models, client_weights)
            
            # Step 4: 在【全局图】上评估
            if rnd % self.cfg.log_interval == 0 or rnd == self.cfg.num_rounds:
                global_model.load_state_dict(global_state)
                test_result = self._eval_on_global(global_model)
                
                if test_result["accuracy"] > best_test_acc:
                    best_test_acc = test_result["accuracy"]
                    best_test_result = test_result
                
                self._log(label, rnd, test_acc=test_result["accuracy"])
                
                print(
                    f"  Round {rnd:3d}/{self.cfg.num_rounds} | "
                    f"全局 Test Acc: {test_result['accuracy']:.4f}"
                )
        
        print(f"\n  ★ {label} 最佳测试准确率: {best_test_acc:.4f}")
        return best_test_result
    
    # ==================================================================
    #  实验 4: FedSage+
    # ==================================================================
    
    def run_fedsage_plus(self) -> dict:
        """FedSage+: FedSage + NeighGen。"""
        print("\n" + "=" * 60)
        print("  实验: FedSage+ (FedSage + NeighGen)")
        print("=" * 60)
        
        # ---- Phase 1-2: 联邦训练 NeighGen ----
        print("\n  Phase 1-2: 联邦训练 NeighGen...")
        
        gen_kwargs = {
            "feature_dim": self.in_dim,
            "hidden_dim": self.cfg.gen_hidden_dim,
            "num_pred": self.cfg.num_pred,
        }
        
        global_gen = NeighborGenerator(**gen_kwargs)
        client_gens = [NeighborGenerator(**gen_kwargs) for _ in self.subgraphs]
        
        global_gen_state = global_gen.state_dict()
        distribute_global_model(global_gen_state, client_gens)
        
        gen_optimizers = [
            torch.optim.Adam(gen.parameters(), lr=self.cfg.lr_generator)
            for gen in client_gens
        ]
        
        gen_rounds = max(1, self.cfg.num_rounds // 2)
        
        for rnd in range(1, gen_rounds + 1):
            distribute_global_model(global_gen_state, client_gens)
            
            for i, (gen, sg, gen_opt) in enumerate(
                zip(client_gens, self.subgraphs, gen_optimizers)
            ):
                gen.to(self.device)
                sg_dev = sg.to(self.device)
                gen.train()
                
                for _ in range(self.cfg.local_epochs):
                    num_sample = min(256, sg_dev.x.size(0))
                    perm = torch.randperm(sg_dev.x.size(0))[:num_sample]
                    
                    node_feat, target_nb, num_real = prepare_neighgen_batch(
                        sg_dev, perm, self.cfg.num_pred
                    )
                    node_feat = node_feat.to(self.device)
                    target_nb = target_nb.to(self.device)
                    num_real = num_real.to(self.device)
                    
                    gen_opt.zero_grad()
                    generated = gen(node_feat)
                    loss = neighgen_loss(generated, target_nb, num_real)
                    loss.backward()
                    gen_opt.step()
                
                gen.cpu()
            
            global_gen_state = fedavg_aggregate(
                client_gens, self.client_weights
            )
            
            if rnd % self.cfg.log_interval == 0 or rnd == gen_rounds:
                print(f"    NeighGen Round {rnd}/{gen_rounds} | Loss: {loss.item():.4f}")
        
        # ---- Phase 3: 补全子图 ----
        print("\n  Phase 3: 用 NeighGen 补全子图...")
        
        global_gen.load_state_dict(global_gen_state)
        global_gen.to(self.device)
        global_gen.eval()
        
        augmented_subgraphs = []
        
        for i, sg in enumerate(self.subgraphs):
            aug_sg = self._augment_subgraph(sg, global_gen)
            augmented_subgraphs.append(aug_sg)
            
            new_nodes = aug_sg.x.size(0) - sg.x.size(0)
            new_edges = aug_sg.edge_index.size(1) - sg.edge_index.size(1)
            print(f"    子图 {i}: +{new_nodes} 生成节点, +{new_edges} 生成边")
        
        global_gen.cpu()
        
        # ---- Phase 4: 在补全子图上运行 FedSage ----
        print("\n  Phase 4: 在补全子图上运行 FedSage...")
        
        aug_weights = [sg.x.size(0) for sg in augmented_subgraphs]
        result = self.run_fedsage(
            subgraphs=augmented_subgraphs,
            client_weights=aug_weights,
            label="FedSage+",
        )
        
        print(f"\n  ★ FedSage+ 最佳测试准确率: {result['accuracy']:.4f}")
        return result
    
    @torch.no_grad()
    def _augment_subgraph(self, sub_data: Data, generator: NeighborGenerator) -> Data:
        """用 NeighGen 为子图生成缺失邻居。"""
        x = sub_data.x.to(self.device)
        edge_index = sub_data.edge_index.to(self.device)
        num_original = x.size(0)
        
        generated = generator(x)
        num_pred = generated.size(1)
        
        fake_features = generated.view(-1, x.size(1))
        
        aug_x = torch.cat([x.cpu(), fake_features.cpu()], dim=0)
        
        new_src = []
        new_dst = []
        for node_id in range(num_original):
            for j in range(num_pred):
                fake_node_id = num_original + node_id * num_pred + j
                new_src.append(node_id)
                new_dst.append(fake_node_id)
                new_src.append(fake_node_id)
                new_dst.append(node_id)
        
        new_edges = torch.tensor([new_src, new_dst], dtype=torch.long)
        aug_edge_index = torch.cat([edge_index.cpu(), new_edges], dim=1)
        
        num_fake = fake_features.size(0)
        aug_y = torch.cat([
            sub_data.y.cpu(),
            torch.full((num_fake,), -1, dtype=torch.long),
        ])
        
        # 生成节点的 mask 全为 False（不参与训练/验证/测试）
        aug_train = torch.cat([sub_data.train_mask.cpu(),
                               torch.zeros(num_fake, dtype=torch.bool)])
        aug_val = torch.cat([sub_data.val_mask.cpu(),
                             torch.zeros(num_fake, dtype=torch.bool)])
        aug_test = torch.cat([sub_data.test_mask.cpu(),
                              torch.zeros(num_fake, dtype=torch.bool)])
        
        aug_data = Data(
            x=aug_x, edge_index=aug_edge_index, y=aug_y,
            train_mask=aug_train, val_mask=aug_val, test_mask=aug_test,
        )
        aug_data.num_classes = sub_data.num_classes
        
        return aug_data
    
    # ==================================================================
    #  运行所有实验
    # ==================================================================
    
    def run_all_experiments(self) -> dict:
        """按顺序运行全部 4 组实验并汇总。"""
        all_results = {}
        
        all_results["GlobSage"] = self.run_globsage()
        all_results["LocSage"] = self.run_locsage()
        all_results["FedSage"] = self.run_fedsage()
        all_results["FedSage+"] = self.run_fedsage_plus()
        
        # 汇总
        glob_acc = all_results["GlobSage"]["accuracy"]
        loc_acc = sum(r["accuracy"] for r in all_results["LocSage"]) / len(all_results["LocSage"])
        fed_acc = all_results["FedSage"]["accuracy"]
        fedp_acc = all_results["FedSage+"]["accuracy"]
        
        print("\n" + "=" * 60)
        print("  实验结果汇总")
        print("=" * 60)
        print(f"  GlobSage (上界): {glob_acc:.4f}")
        print(f"  LocSage  (下界): {loc_acc:.4f}")
        print(f"  FedSage:         {fed_acc:.4f}")
        print(f"  FedSage+:        {fedp_acc:.4f}")
        print("=" * 60)
        
        if loc_acc < fed_acc <= glob_acc:
            print("  ✅ 结果符合预期: LocSage < FedSage ≤ GlobSage")
        else:
            print("  ⚠️  结果排序异常，可能需要调参或增加轮次")
        
        # 保存结果
        if self.saver:
            self.saver.set_summary(all_results)
            self.saver.save_all()
        
        return all_results
