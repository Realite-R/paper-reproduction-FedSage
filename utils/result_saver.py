"""FedSage 论文复现 —— 实验结果保存"""
import os, json, csv, time
from datetime import datetime

class ResultSaver:
    def __init__(self, cfg):
        self.cfg = cfg
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{timestamp}_{cfg.dataset}_M{cfg.num_owners}"
        self.exp_dir = os.path.join(cfg.result_path, folder_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.training_log = []
        self.summary = {}
        print(f"  实验结果将保存到: {self.exp_dir}")
    
    def log_round(self, experiment, round_num, **metrics):
        entry = {"experiment": experiment, "round": round_num, "timestamp": time.time()}
        entry.update(metrics)
        self.training_log.append(entry)
    
    def set_summary(self, all_results):
        self.summary = all_results
    
    def save_all(self):
        self._save_config()
        self._save_summary()
        self._save_training_log()
        self._save_report()
        self._save_training_curves()
        print(f"\n  📁 所有结果已保存到: {self.exp_dir}")
    
    def _save_config(self):
        d = {k: getattr(self.cfg, k) for k in [
            "dataset","num_owners","num_rounds","local_epochs","hidden_dim",
            "num_layers","dropout","lr_classifier","weight_decay","num_pred",
            "gen_hidden_dim","lr_generator","seed","device","train_ratio","val_ratio"
        ]}
        with open(os.path.join(self.exp_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
    
    def _save_summary(self):
        s = {}
        for k, v in self.summary.items():
            if isinstance(v, dict):
                s[k] = {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
            elif isinstance(v, list):
                s[k] = [{kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in item.items()} if isinstance(item, dict) else item for item in v]
            else:
                s[k] = v
        with open(os.path.join(self.exp_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
    
    def _save_training_log(self):
        if not self.training_log: return
        all_keys = set()
        for e in self.training_log: all_keys.update(e.keys())
        fixed = ["experiment", "round"]
        other = sorted(all_keys - set(fixed) - {"timestamp"})
        fieldnames = fixed + other
        with open(os.path.join(self.exp_dir, "training_log.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for e in self.training_log:
                row = {k: round(v, 4) if isinstance(v, float) else v for k, v in e.items()}
                w.writerow(row)
    
    def _save_report(self):
        lines = ["="*65, "  FedSage 论文复现 —— 实验报告", "="*65,
                 f"  日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 f"  数据集: {self.cfg.dataset}  M={self.cfg.num_owners}",
                 f"  轮次: {self.cfg.num_rounds}  lr: {self.cfg.lr_classifier}", ""]
        for name in ["GlobSage", "LocSage", "FedSage", "FedSage+"]:
            if name not in self.summary: continue
            val = self.summary[name]
            if isinstance(val, list):
                accs = [r.get("accuracy",0) for r in val if isinstance(r,dict)]
                avg = sum(accs)/len(accs) if accs else 0
                lines.append(f"  {name:<15} {avg:.4f}")
            elif isinstance(val, dict):
                lines.append(f"  {name:<15} {val.get('accuracy',0):.4f}")
        lines.extend(["", "="*65])
        with open(os.path.join(self.exp_dir, "report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def _save_training_curves(self):
        if not self.training_log: return
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        except ImportError: return
        exps = {}
        for e in self.training_log:
            name = e["experiment"]
            if name not in exps: exps[name] = {"r":[], "a":[]}
            if "test_acc" in e:
                exps[name]["r"].append(e["round"])
                exps[name]["a"].append(e["test_acc"])
        colors = {"GlobSage":"#2ecc71","LocSage":"#e74c3c","FedSage":"#3498db","FedSage+":"#9b59b6"}
        fig, ax = plt.subplots(figsize=(10,6))
        for n, d in exps.items():
            if d["a"]: ax.plot(d["r"], d["a"], label=n, color=colors.get(n,"#333"), linewidth=2)
        ax.set_xlabel("Round/Epoch"); ax.set_ylabel("Test Accuracy"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title(f"FedSage — {self.cfg.dataset.upper()}, M={self.cfg.num_owners}")
        plt.tight_layout()
        fig.savefig(os.path.join(self.exp_dir, "training_curves.png"), dpi=150)
        plt.close(fig)
