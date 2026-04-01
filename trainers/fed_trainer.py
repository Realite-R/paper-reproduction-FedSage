"""
FedSage 论文复现 —— 联邦训练编排器 (v5 - 对齐原仓库)
=======================================================

v5 修复清单：
  1. FedAvg 改为等权（1/M）聚合
  2. NeighGen 训练：隐藏节点 → 找缺失邻居 → 三部分损失
  3. 子图补全：按 dGen 预测的缺失数量生成（非固定 num_pred）
  4. 超参对齐：gen_rounds=20, num_rounds=20, local_epochs=1
"""
import copy
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data

from models.sage import GraphSageClassifier
from models.neighgen import (
    NeighborGenerator,
    neighgen_loss,
    prepare_neighgen_batch,
)
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

        assert hasattr(global_data, "train_mask") and global_data.train_mask is not None, \
            "全局图必须先做 train/val/test 划分！"

    def _make_model_kwargs(self):
        return {
            "in_dim": self.in_dim,
            "hidden_dim": self.cfg.hidden_dim,
            "out_dim": self.out_dim,
            "num_layers": self.cfg.num_layers,
            "dropout": self.cfg.dropout,
        }

    def _log(self, experiment, round_num, **metrics):
        if self.saver:
            self.saver.log_round(experiment, round_num, **metrics)

    def _eval_on_global(self, model) -> dict:
        """在全局图的测试节点上评估。"""
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
        print("\n" + "=" * 60)
        print("  实验: GlobSage (全局图集中训练 — 性能上界)")
        print("=" * 60)

        model = GraphSageClassifier(**self._make_model_kwargs())
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
                    self.global_data.val_mask, self.device,
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

    def run_fedsage(self, subgraphs=None, label="FedSage") -> dict:
        if subgraphs is None:
            subgraphs = self.subgraphs

        print("\n" + "=" * 60)
        print(f"  实验: {label} (FedAvg + GraphSage)")
        print("=" * 60)

        model_kwargs = self._make_model_kwargs()
        global_model = GraphSageClassifier(**model_kwargs)
        global_state = global_model.state_dict()

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
            # 下发全局参数
            for trainer in client_trainers:
                trainer.reload_model_params(global_state)

            # 本地训练
            for trainer in client_trainers:
                trainer.train_multiple_epochs(self.cfg.local_epochs)

            # FedAvg 聚合（等权，对齐原仓库）
            global_state = fedavg_aggregate(
                client_models, mode=self.cfg.fedavg_weight,
            )

            # 评估
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
        """
        FedSage+: FedSage + NeighGen。

        对齐原仓库的完整流程：
          Phase 1-2: 联邦训练 NeighGen（隐藏节点 → 找缺失邻居 → 三部分损失）
          Phase 3:   用训练好的 NeighGen 补全子图（按 dGen 预测数量，非固定 num_pred）
          Phase 4:   在补全子图上运行 FedSage
        """
        print("\n" + "=" * 60)
        print("  实验: FedSage+ (FedSage + NeighGen)")
        print("=" * 60)

        # ---- Phase 1-2: 联邦训练 NeighGen ----
        print("\n  Phase 1-2: 联邦训练 NeighGen（隐藏节点语义）...")

        gen_kwargs = {
            "feature_dim": self.in_dim,
            "hidden_dim": self.cfg.gen_hidden_dim,
            "num_pred": self.cfg.num_pred,
            "num_classes": self.out_dim,
        }

        global_gen = NeighborGenerator(**gen_kwargs)
        client_gens = [NeighborGenerator(**gen_kwargs) for _ in self.subgraphs]

        global_gen_state = global_gen.state_dict()
        distribute_global_model(global_gen_state, client_gens)

        gen_optimizers = [
            torch.optim.Adam(gen.parameters(), lr=self.cfg.lr_generator)
            for gen in client_gens
        ]

        rng = np.random.RandomState(self.cfg.seed)

        for rnd in range(1, self.cfg.gen_rounds + 1):
            # 下发全局 NeighGen 参数
            distribute_global_model(global_gen_state, client_gens)

            round_losses = []

            for i, (gen, sg, gen_opt) in enumerate(
                zip(client_gens, self.subgraphs, gen_optimizers)
            ):
                gen.to(self.device)
                sg_dev = sg.to(self.device)
                gen.train()

                for _ in range(self.cfg.local_epochs):
                    # ★ 关键修复：使用隐藏节点语义准备 batch
                    node_feat, target_deg, target_nb, target_labels = \
                        prepare_neighgen_batch(
                            sg_dev,
                            hide_portion=self.cfg.hide_portion,
                            num_pred=self.cfg.num_pred,
                            rng=rng,
                        )

                    node_feat = node_feat.to(self.device)
                    target_deg = target_deg.to(self.device)
                    target_nb = target_nb.to(self.device)
                    target_labels = target_labels.to(self.device)

                    gen_opt.zero_grad()
                    pred_deg, gen_features, class_logits = gen(node_feat)

                    # ★ 三部分损失
                    loss, loss_dict = neighgen_loss(
                        pred_degree=pred_deg,
                        gen_features=gen_features,
                        class_logits=class_logits,
                        target_degree=target_deg,
                        target_neighbors=target_nb,
                        target_labels=target_labels,
                    )

                    loss.backward()
                    gen_opt.step()

                round_losses.append(loss_dict)
                gen.cpu()

            # FedAvg 聚合 NeighGen（等权）
            global_gen_state = fedavg_aggregate(
                client_gens, mode=self.cfg.fedavg_weight,
            )

            if rnd % self.cfg.log_interval == 0 or rnd == self.cfg.gen_rounds:
                avg_loss = np.mean([d["total"] for d in round_losses])
                avg_deg = np.mean([d["degree"] for d in round_losses])
                avg_feat = np.mean([d["feature"] for d in round_losses])
                avg_cls = np.mean([d["class"] for d in round_losses])
                print(
                    f"    NeighGen Round {rnd:2d}/{self.cfg.gen_rounds} | "
                    f"Total: {avg_loss:.4f} "
                    f"(deg: {avg_deg:.4f}, feat: {avg_feat:.4f}, cls: {avg_cls:.4f})"
                )

        # ---- Phase 3: 用 NeighGen 补全子图 ----
        print("\n  Phase 3: 用 NeighGen 补全子图（按 dGen 预测数量）...")

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

        result = self.run_fedsage(
            subgraphs=augmented_subgraphs,
            label="FedSage+",
        )

        print(f"\n  ★ FedSage+ 最佳测试准确率: {result['accuracy']:.4f}")
        return result

    @torch.no_grad()
    def _augment_subgraph(self, sub_data: Data, generator: NeighborGenerator) -> Data:
        """
        用 NeighGen 为子图生成缺失邻居。

        ★ 关键修复（对齐原仓库）：
        之前：每个节点固定生成 num_pred 个邻居
        现在：用 dGen 预测每个节点的缺失数量，只生成对应数量的邻居
        """
        x = sub_data.x.to(self.device)
        edge_index = sub_data.edge_index.to(self.device)
        num_original = x.size(0)

        # 用 NeighGen 的两个输出
        pred_degree, gen_features, _ = generator(x)

        # pred_degree: [N, 1]，四舍五入到整数，截断到 [0, num_pred]
        pred_counts = pred_degree.squeeze(-1).round().long()
        pred_counts = pred_counts.clamp(min=0, max=self.cfg.num_pred)

        # 收集生成的节点和边
        new_features_list = []
        new_src = []
        new_dst = []
        fake_node_offset = num_original

        for node_id in range(num_original):
            n_gen = pred_counts[node_id].item()
            if n_gen == 0:
                continue

            # 取前 n_gen 个生成的邻居特征
            node_gen_feats = gen_features[node_id, :n_gen, :]  # [n_gen, feat_dim]
            new_features_list.append(node_gen_feats)

            for j in range(n_gen):
                fake_id = fake_node_offset
                new_src.append(node_id)
                new_dst.append(fake_id)
                new_src.append(fake_id)
                new_dst.append(node_id)
                fake_node_offset += 1

        if len(new_features_list) == 0:
            # 没有需要生成的邻居，返回原图
            return sub_data

        fake_features = torch.cat(new_features_list, dim=0)  # [total_fake, feat_dim]
        num_fake = fake_features.size(0)

        # 拼接特征
        aug_x = torch.cat([x.cpu(), fake_features.cpu()], dim=0)

        # 拼接边
        new_edges = torch.tensor([new_src, new_dst], dtype=torch.long)
        aug_edge_index = torch.cat([edge_index.cpu(), new_edges], dim=1)

        # 拼接标签（生成节点标签设为 -1）
        aug_y = torch.cat([
            sub_data.y.cpu(),
            torch.full((num_fake,), -1, dtype=torch.long),
        ])

        # 生成节点的 mask 全为 False
        aug_train = torch.cat([
            sub_data.train_mask.cpu(),
            torch.zeros(num_fake, dtype=torch.bool),
        ])
        aug_val = torch.cat([
            sub_data.val_mask.cpu(),
            torch.zeros(num_fake, dtype=torch.bool),
        ])
        aug_test = torch.cat([
            sub_data.test_mask.cpu(),
            torch.zeros(num_fake, dtype=torch.bool),
        ])

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
        all_results = {}

        all_results["GlobSage"] = self.run_globsage()
        all_results["LocSage"] = self.run_locsage()
        all_results["FedSage"] = self.run_fedsage()
        all_results["FedSage+"] = self.run_fedsage_plus()

        # 汇总
        glob_acc = all_results["GlobSage"]["accuracy"]
        loc_acc = sum(r["accuracy"] for r in all_results["LocSage"]) / len(
            all_results["LocSage"]
        )
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

        if self.saver:
            self.saver.set_summary(all_results)
            self.saver.save_all()

        return all_results
