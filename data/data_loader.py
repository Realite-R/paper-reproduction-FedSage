"""FedSage 论文复现 —— 数据加载模块"""
import os
import sys
import time
import urllib.request
import ssl
import torch
from torch_geometric.data import Data

PLANETOID_URL = "https://github.com/kimiyoung/planetoid/raw/master/data"
PLANETOID_FILES = [
    "ind.{name}.x", "ind.{name}.tx", "ind.{name}.allx",
    "ind.{name}.y", "ind.{name}.ty", "ind.{name}.ally",
    "ind.{name}.graph", "ind.{name}.test.index",
]


def _download_file(url, save_path, timeout=30):
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
            data = response.read()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def _ensure_planetoid_downloaded(name, data_path):
    name_cap = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[name]
    raw_dir = os.path.join(data_path, name_cap, "raw")
    all_files = [f.format(name=name) for f in PLANETOID_FILES]
    missing = [f for f in all_files if not os.path.exists(os.path.join(raw_dir, f))]
    if not missing:
        print(f"  数据集文件已存在于 {raw_dir}")
        return True
    print(f"  正在下载 {name_cap} 数据集 ({len(missing)} 个文件)...")
    os.makedirs(raw_dir, exist_ok=True)
    success = 0
    for fn in missing:
        url = f"{PLANETOID_URL}/{fn}"
        for attempt in range(2):
            if _download_file(url, os.path.join(raw_dir, fn)):
                success += 1
                print(f"    ✅ {fn}")
                break
            time.sleep(1)
        else:
            print(f"    ❌ {fn}")
    return success == len(missing)


def load_dataset(dataset_name, data_path):
    from torch_geometric.datasets import Planetoid, Amazon

    dataset_name = dataset_name.lower()

    if dataset_name in ["cora", "citeseer", "pubmed"]:
        name_map = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
        if not _ensure_planetoid_downloaded(dataset_name, data_path):
            print(f"\n  ⚠️ 下载失败，请设置代理: set https_proxy=http://127.0.0.1:7890")
            sys.exit(1)
        dataset = Planetoid(root=data_path, name=name_map[dataset_name])
    elif dataset_name in ["amazon-computers", "computers"]:
        try:
            dataset = Amazon(root=data_path, name="Computers")
        except Exception:
            print(f"\n  ⚠️ Amazon 数据集下载失败，请设置代理后重试。")
            sys.exit(1)
    else:
        raise ValueError(f"不支持的数据集: '{dataset_name}'")

    data = dataset[0]
    data.num_classes = dataset.num_classes
    return data


def get_dataset_info(data):
    from torch_geometric.utils import degree

    deg = degree(data.edge_index[0], num_nodes=data.x.size(0))
    return {
        "num_nodes": data.x.size(0),
        "num_edges": data.edge_index.size(1),
        "num_features": data.x.size(1),
        "num_classes": data.num_classes,
        "avg_degree": deg.float().mean().item(),
    }


def print_dataset_info(data, dataset_name):
    info = get_dataset_info(data)
    print(f"\n{'=' * 50}")
    print(f"  数据集: {dataset_name}")
    print(f"{'=' * 50}")
    print(f"  节点数:   {info['num_nodes']}")
    print(f"  边数:     {info['num_edges']}")
    print(f"  特征维度: {info['num_features']}")
    print(f"  类别数:   {info['num_classes']}")
    print(f"  平均度:   {info['avg_degree']:.2f}")
    print(f"{'=' * 50}\n")
