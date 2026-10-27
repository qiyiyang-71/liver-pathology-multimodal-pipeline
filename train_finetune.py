# -*- coding: utf-8 -*-
"""肝脏病理图像微调训练脚本（云端 GPU 版）
任务（多任务共享主干网络）：
  1. 炎症分级 G 五分类（G0-G4，仅光镜 HE 图像）
  2. 显著纤维化筛查 S>=3 二分类
数据：建模清单_图像级.csv（已按患者划分 train/test，无患者泄漏）
评估：报告级聚合（同一报告多张图像概率平均后再判定）
用法：
  HF_ENDPOINT=https://hf-mirror.com python train_finetune.py --data_root /root/data --epochs 30
"""
import os, json, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "WenQuanYi Zen Hei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HE_TYPES = ["全套餐", "炎症评估", "疑难肝病"]
G_CLASSES = ["G0", "G1", "G2", "G3", "G4"]
MODEL_NAME = "vinid/plip"


def resolve_model(data_root):
    """优先使用本地模型文件夹（plip模型/），避免在线下载失败"""
    local = os.path.join(data_root, "plip模型")
    if os.path.isdir(local):
        print(f"使用本地模型文件夹: {local}")
        return local
    return MODEL_NAME


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class LiverDataset(Dataset):
    def __init__(self, df, data_root, proc, train=True):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.proc = proc
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(self.data_root, r["图像路径"])).convert("RGB")
        if self.train:  # 训练时随机裁剪/翻转做数据增强
            img = img.resize((256, 256))
            img = img.crop((random.randint(0, 32), random.randint(0, 32),
                            random.randint(0, 32) + 224, random.randint(0, 32) + 224))
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = img.resize((224, 224))
        x = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        g = -1 if pd.isna(r["炎症分级"]) else int(r["炎症分级"])
        s = -1 if pd.isna(r["纤维化分期下限"]) else (1 if int(r["纤维化分期下限"]) >= 3 else 0)
        return x, g, s, r["研究编号"]


class MultiTaskModel(nn.Module):
    def __init__(self, model_name, feat_dim=768):  # PLiP vision_model 的 pooler 输出为 768 维
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name).vision_model
        self.head_g = nn.Linear(feat_dim, 5)
        self.head_s = nn.Linear(feat_dim, 2)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        feat = out.pooler_output
        return self.head_g(feat), self.head_s(feat), feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--lr_backbone", type=float, default=3e-6)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--no_freeze", action="store_true", help="放开主干网络微调（小数据不建议）")
    ap.add_argument("--out", default="训练输出")
    args = ap.parse_args()
    set_seed()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(os.path.join(args.data_root, "建模清单_图像级.csv"))
    df = df[df["报告类型"].isin(HE_TYPES)].reset_index(drop=True)
    model_name = resolve_model(args.data_root)
    proc = AutoProcessor.from_pretrained(model_name)
    tr_df = df[df["数据集划分"] == "train"].reset_index(drop=True)
    te_df = df[df["数据集划分"] == "test"].reset_index(drop=True)
    print(f"光镜图像: 训练 {len(tr_df)} 张 / 测试 {len(te_df)} 张; device={device}")

    model = MultiTaskModel(model_name).to(device)
    head_params = list(model.head_g.parameters()) + list(model.head_s.parameters())
    if not args.no_freeze:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("模式: 冻结主干，只训练分类头（小数据推荐）")
    else:
        print("模式: 放开主干微调")
    opt = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=1e-4)
    steps = args.epochs * (len(tr_df) // args.batch + 1)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)

    # 类别加权：根据训练集标签分布计算权重，缓解 G2 / <S3 一边倒
    def class_weights(labels, n_cls):
        cnt = np.bincount(labels, minlength=n_cls).astype(np.float32)
        w = cnt.sum() / np.maximum(cnt, 1) / n_cls
        return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

    g_tr = tr_df["炎症分级"].dropna().astype(int).to_numpy()
    w_g = class_weights(g_tr, 5)
    s_tr = (tr_df["纤维化分期下限"].dropna().astype(int) >= 3).astype(int).to_numpy()
    w_s = class_weights(s_tr, 2)
    ce = nn.CrossEntropyLoss(weight=w_g, label_smoothing=args.label_smoothing)
    ce_s = nn.CrossEntropyLoss(weight=w_s, label_smoothing=args.label_smoothing)
    print(f"G 类权重: {[round(float(x),2) for x in w_g]}; S 类权重: {[round(float(x),2) for x in w_s]}")

    tr_loader = DataLoader(LiverDataset(tr_df, args.data_root, proc, True),
                           batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)

    for ep in range(args.epochs):
        model.train(); tot = n = 0
        for x, g, s, _ in tr_loader:
            x = x.to(device, non_blocking=True)
            pg, ps, _ = model(x)
            loss = 0
            m = g >= 0
            if m.any(): loss = loss + ce(pg[m], g[m].to(device))
            m = s >= 0
            if m.any(): loss = loss + ce_s(ps[m], s[m].to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); n += 1
        print(f"epoch {ep+1}/{args.epochs}  loss={tot/max(n,1):.4f}", flush=True)
        torch.save(model.state_dict(), os.path.join(args.out, "模型最新.pt"))
    print("训练完成，开始测试集评估...")

    # ---- 测试集评估（报告级聚合）----
    model.eval()
    te_loader = DataLoader(LiverDataset(te_df, args.data_root, proc, False),
                           batch_size=64, shuffle=False, num_workers=4)
    pg_all, ps_all, reps = [], [], []
    with torch.no_grad():
        for x, g, s, rp in te_loader:
            x = x.to(device)
            pg, ps, _ = model(x)
            pg_all.append(pg.cpu()); ps_all.append(ps.cpu())
            reps += list(rp)
    PG = torch.cat(pg_all).softmax(1).numpy()
    PS = torch.cat(ps_all).softmax(1).numpy()

    def agg(proba, df_te, kind):
        """按研究编号聚合平均概率; kind='g' 或 's'"""
        out = {}
        for rep in df_te["研究编号"].unique():
            idx = [i for i, r in enumerate(reps) if r == rep]
            meanp = proba[idx].mean(0)
            out[rep] = meanp
        return out

    res = {}
    te_g = te_df.drop_duplicates("研究编号").set_index("研究编号")
    agg_g = agg(PG, te_df, "g")
    yt_g = np.array([int(te_g.loc[r, "炎症分级"]) for r in agg_g])
    yp_g = np.array([int(v.argmax()) for v in agg_g.values()])
    res["G分级"] = {"报告级准确率": round(float(accuracy_score(yt_g, yp_g)), 3),
                "宏平均F1": round(float(f1_score(yt_g, yp_g, average="macro")), 3)}

    agg_s = agg(PS, te_df, "s")
    yt_s = np.array([1 if int(te_g.loc[r, "纤维化分期下限"]) >= 3 else 0 for r in agg_s])
    yp_s = np.array([int(v.argmax()) for v in agg_s.values()])
    yprob_s = np.array([v[1] for v in agg_s.values()])
    res["S>=3筛查"] = {"报告级准确率": round(float(accuracy_score(yt_s, yp_s)), 3),
                  "F1": round(float(f1_score(yt_s, yp_s)), 3),
                  "AUC": round(float(roc_auc_score(yt_s, yprob_s)), 3)}

    with open(os.path.join(args.out, "测试结果.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    # 混淆矩阵
    for name, yt, yp, labels in [("G分级", yt_g, yp_g, G_CLASSES),
                                 ("S>=3筛查", yt_s, yp_s, ["<S3", ">=S3"])]:
        cm = confusion_matrix(yt, yp, labels=list(range(len(labels))))
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("预测"); ax.set_ylabel("真实"); ax.set_title(f"微调模型 {name} 混淆矩阵")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, shrink=0.8); fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"混淆矩阵_{name}.png"), dpi=150)
    torch.save(model.state_dict(), os.path.join(args.out, "模型最终.pt"))
    print("全部结果已保存到", args.out)


if __name__ == "__main__":
    main()
