# -*- coding: utf-8 -*-
"""分级 V2 + 补丁：在 V2 基础上额外保存每个报告（研究编号）的预测 G/S 标签
供 A5 条件生成使用（image -> 分级模型 -> 预测标签 -> 生成报告 的真实管线）。
输出新增：训练输出_分级V2/报告预测.json  {研究编号: {G5, G2, S3, ...}}
用法：
  HF_ENDPOINT=https://hf-mirror.com python 分级V2_微调PLiP_保存预测.py --data_root . --epochs 12 --batch 32
（原 V2 逻辑完全不变，仅在最后汇总时多存一个文件）
"""
import os, json, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
from transformers import AutoModel, AutoProcessor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_fscore_support, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HE_TYPES = ["全套餐", "炎症评估", "疑难肝病"]
MODEL_NAME = "vinid/plip"
OUT_DIR = "训练输出_分级V2"


def resolve_model(data_root):
    local = os.path.join(data_root, "plip模型")
    if os.path.isdir(local):
        print(f"使用本地模型文件夹: {local}")
        return local
    return MODEL_NAME


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def mid_label(lo, hi):
    if pd.isna(lo):
        return np.nan
    if pd.isna(hi):
        hi = lo
    return int(np.floor((lo + hi) / 2 + 0.5))  # 中点四舍五入，.5 向上


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
        if self.train:
            img = img.resize((256, 256))
            img = img.crop((random.randint(0, 32), random.randint(0, 32),
                            random.randint(0, 32) + 224, random.randint(0, 32) + 224))
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
        else:
            img = img.resize((224, 224))
        x = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        g5 = -1 if pd.isna(r["G标签"]) else int(r["G标签"])
        g2 = -1 if pd.isna(r["G标签"]) else (1 if int(r["G标签"]) >= 2 else 0)
        s3 = -1 if pd.isna(r["S中点"]) else (1 if int(r["S中点"]) >= 3 else 0)
        return x, g5, g2, s3, r["研究编号"]


def collate(batch):
    x = torch.stack([b[0] for b in batch])
    g5 = torch.tensor([b[1] for b in batch])
    g2 = torch.tensor([b[2] for b in batch])
    s3 = torch.tensor([b[3] for b in batch])
    return x, g5, g2, s3, [b[4] for b in batch]


class MultiTaskNet(nn.Module):
    """共享 PLiP 视觉编码器 + 三个线性头"""
    def __init__(self, model_path):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_path).vision_model
        hid = self.enc.config.hidden_size
        self.head_g5 = nn.Linear(hid, 5)
        self.head_g2 = nn.Linear(hid, 2)
        self.head_s3 = nn.Linear(hid, 2)

    def forward(self, x):
        emb = self.enc(pixel_values=x).pooler_output  # [B, hidden]
        return self.head_g5(emb), self.head_g2(emb), self.head_s3(emb)


def class_weights(labels, n_cls, device):
    """N/(C*n_c) 重加权；缺失类给权重 0"""
    w = torch.zeros(n_cls, device=device)
    for c in range(n_cls):
        n_c = (labels == c).sum().item()
        if n_c > 0:
            w[c] = len(labels) / (n_cls * n_c)
    return w


def masked_ce(logits, targets, weights):
    m = targets >= 0
    if m.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return nn.functional.cross_entropy(logits[m], targets[m], weight=weights)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    out = {"g5": [], "g2": [], "s3": [], "rep": [], "g5y": [], "g2y": [], "s3y": []}
    for x, g5, g2, s3, reps in loader:
        x = x.to(device)
        lg5, lg2, ls3 = model(x)
        out["g5"].append(lg5.softmax(1).cpu())
        out["g2"].append(lg2.softmax(1).cpu())
        out["s3"].append(ls3.softmax(1).cpu())
        out["rep"].extend(reps)
        out["g5y"].extend(g5.tolist()); out["g2y"].extend(g2.tolist()); out["s3y"].extend(s3.tolist())
    for k in ["g5", "g2", "s3"]:
        out[k] = torch.cat(out[k]).numpy()
    for k in ["g5y", "g2y", "s3y"]:
        out[k] = np.array(out[k])
    return out


def report_level(probs, labels, reps):
    """同报告多图概率平均；标签取首值（同报告内一致）"""
    _, inv = np.unique(reps, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    summed = np.zeros((len(counts), probs.shape[1]))
    np.add.at(summed, inv, probs)
    p = summed / counts[:, None]
    y = labels[np.unique(inv, return_index=True)[1]]
    return p, y


def youden_threshold(p1, y):
    """在验证集（报告级）上选使 敏感度+特异度-1 最大的判定门槛（约登指数）"""
    m = y >= 0
    p1, y = p1[m], y[m]
    if len(np.unique(y)) < 2:
        return 0.5
    best_thr, best_j = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (p1 >= thr).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        j = tp / (tp + fn) + tn / (tn + fp) - 1
        if j > best_j:
            best_j, best_thr = j, thr
    return float(best_thr)


def evaluate(name, probs, labels, reps, names, focus=None, thr_per_sample=None):
    # 报告级聚合：同报告多图概率平均，标签/阈值取首值（同报告内一致）
    _, inv = np.unique(reps, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    summed = np.zeros((len(counts), probs.shape[1]))
    np.add.at(summed, inv, probs)
    p = summed / counts[:, None]
    first = np.unique(inv, return_index=True)[1]
    y = labels[first]
    if thr_per_sample is not None:
        thr_rep = np.asarray(thr_per_sample, dtype=float)[first]
    m = y >= 0
    p, y = p[m], y[m]
    if thr_per_sample is not None:
        pred = (p[:, 1] >= thr_rep[m]).astype(int)
    else:
        pred = p.argmax(1)
    acc = accuracy_score(y, pred)
    mf1 = f1_score(y, pred, average="macro", labels=list(range(len(names))), zero_division=0)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, pred, labels=list(range(len(names))), zero_division=0)
    try:
        auc = roc_auc_score(y, p[:, 1]) if len(names) == 2 else roc_auc_score(
            y, p, multi_class="ovr", labels=list(range(len(names))))
    except ValueError:
        auc = float("nan")
    # 患者级 bootstrap 95% CI
    rng = np.random.default_rng(42)
    accs, mf1s, sens = [], [], []
    for _ in range(1000):
        pick = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[pick])) < 2:
            continue
        accs.append(accuracy_score(y[pick], pred[pick]))
        mf1s.append(f1_score(y[pick], pred[pick], average="macro",
                             labels=list(range(len(names))), zero_division=0))
        if focus is not None:
            tp = ((pred[pick] == focus) & (y[pick] == focus)).sum()
            fn = ((pred[pick] != focus) & (y[pick] == focus)).sum()
            sens.append(tp / (tp + fn) if tp + fn > 0 else np.nan)
    def ci_or_nan(vals):
        return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))] if len(vals) else [float("nan"), float("nan")]
    res = {"acc": acc, "macro_f1": mf1, "auc": auc,
           "per_class": {names[i]: [float(pr[i]), float(rc[i]), float(f1[i])] for i in range(len(names))},
           "ci_acc": ci_or_nan(accs),
           "ci_macro_f1": ci_or_nan(mf1s)}
    if focus is not None:
        res["sensitivity_focus"] = float(rc[focus])
        res["ci_sens"] = ci_or_nan(sens)
    return res, y, pred


def plot_cm(name, y, pred, names, path):
    cm = confusion_matrix(y, pred, labels=list(range(len(names))))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names); ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("预测"); ax.set_ylabel("真实"); ax.set_title(f"{name} 混淆矩阵（5折汇集）")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr_enc", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    args = ap.parse_args()
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = os.path.join(args.data_root, "建模清单_图像级.csv")
    df = pd.read_csv(csv_path)
    df = df[df["报告类型"].isin(HE_TYPES)].reset_index(drop=True)
    df["S中点"] = [mid_label(lo, hi) for lo, hi in zip(df["纤维化分期下限"], df["纤维化分期上限"])]
    df["G标签"] = df["炎症分级"]
    print(f"光镜图像总数: {len(df)}，患者数: {df['患者组编号'].nunique()}")

    model_path = resolve_model(args.data_root)
    proc = AutoProcessor.from_pretrained(model_path)

    gkf = GroupKFold(n_splits=5)
    oof = {"g5": [], "g2": [], "s3": []}  # 汇集各折测试预测
    oof_rep, oof_y = {"g5": [], "g2": [], "s3": []}, {"g5": [], "g2": [], "s3": []}
    oof_fold = {"g5": [], "g2": [], "s3": []}   # 每条样本属于第几折（用该折的门槛）
    oof_thr = {"g2": [], "s3": []}              # 每折在验证集上选出的门槛
    fold_log = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, groups=df["患者组编号"]), 1):
        df_tr_all, df_te = df.iloc[tr_idx], df.iloc[te_idx].reset_index(drop=True)
        # 训练患者里再切 15% 做早停验证集
        rng = np.random.default_rng(42 + fold)
        patients = df_tr_all["患者组编号"].unique()
        val_p = set(rng.choice(patients, size=max(1, int(len(patients) * 0.15)), replace=False))
        df_tr = df_tr_all[~df_tr_all["患者组编号"].isin(val_p)].reset_index(drop=True)
        df_va = df_tr_all[df_tr_all["患者组编号"].isin(val_p)].reset_index(drop=True)
        print(f"\n===== Fold {fold}/5  train {len(df_tr)}  val {len(df_va)}  test {len(df_te)} =====")

        dl_tr = DataLoader(LiverDataset(df_tr, args.data_root, proc, True), batch_size=args.batch,
                           shuffle=True, num_workers=2, collate_fn=collate)
        dl_va = DataLoader(LiverDataset(df_va, args.data_root, proc, False), batch_size=args.batch,
                           shuffle=False, num_workers=2, collate_fn=collate)

        model = MultiTaskNet(model_path).to(device)
        opt = torch.optim.AdamW([
            {"params": model.enc.parameters(), "lr": args.lr_enc},
            {"params": list(model.head_g5.parameters()) + list(model.head_g2.parameters())
                       + list(model.head_s3.parameters()), "lr": args.lr_head},
        ], weight_decay=1e-4)
        steps = max(1, len(dl_tr) * args.epochs)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

        # 类别权重（用整个训练集标签算一次）
        g5_lab = df_tr.loc[df_tr["G标签"].notna(), "G标签"].astype(int).to_numpy()
        g2_lab = (df_tr.loc[df_tr["G标签"].notna(), "G标签"] >= 2).astype(int).to_numpy()
        s3_lab = (df_tr.loc[df_tr["S中点"].notna(), "S中点"] >= 3).astype(int).to_numpy()
        w_g5 = class_weights(torch.tensor(g5_lab), 5, device)
        w_g2 = class_weights(torch.tensor(g2_lab), 2, device)
        w_s3 = class_weights(torch.tensor(s3_lab), 2, device)

        best_auc, best_state, patience = float("-inf"), None, 0
        for ep in range(1, args.epochs + 1):
            model.train()
            tot = 0.0
            for x, g5, g2, s3, _ in dl_tr:
                x, g5, g2, s3 = x.to(device), g5.to(device), g2.to(device), s3.to(device)
                lg5, lg2, ls3 = model(x)
                loss = (masked_ce(lg5, g5, w_g5) + masked_ce(lg2, g2, w_g2)
                        + masked_ce(ls3, s3, w_s3))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += loss.item()
            # 早停：验证集两个二分类 AUC 的均值
            vp = predict_probs(model, dl_va, device)
            aucs = []
            for k in ["g2", "s3"]:
                m = vp[k + "y"] >= 0
                if len(np.unique(vp[k + "y"][m])) == 2:
                    aucs.append(roc_auc_score(vp[k + "y"][m], vp[k][m][:, 1]))
            va_auc = float(np.mean(aucs)) if aucs else -1
            print(f"  epoch {ep}/{args.epochs} loss={tot/len(dl_tr):.4f} val_auc={va_auc:.3f}")
            if va_auc > best_auc:
                best_auc, best_state, patience = va_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 4:
                    print("  早停")
                    break
        model.load_state_dict(best_state)

        # 在验证集上为两个二分类任务选判定门槛（约登指数）
        vp = predict_probs(model, dl_va, device)
        pv_g2, yv_g2 = report_level(vp["g2"], vp["g2y"], vp["rep"])
        thr_g2 = youden_threshold(pv_g2[:, 1], yv_g2)
        pv_s3, yv_s3 = report_level(vp["s3"], vp["s3y"], vp["rep"])
        thr_s3 = youden_threshold(pv_s3[:, 1], yv_s3)
        print(f"  验证集选门槛: G≥2 thr={thr_g2:.2f}  S≥3 thr={thr_s3:.2f}")

        # 测试（本折未见过的患者）
        dl_te = DataLoader(LiverDataset(df_te, args.data_root, proc, False), batch_size=args.batch,
                           shuffle=False, num_workers=2, collate_fn=collate)
        tp = predict_probs(model, dl_te, device)
        for k in ["g5", "g2", "s3"]:
            oof[k].append(tp[k]); oof_rep[k].extend(tp["rep"]); oof_y[k].append(tp[k + "y"])
            oof_fold[k].extend([fold] * len(tp["rep"]))
        oof_thr["g2"].append(thr_g2)
        oof_thr["s3"].append(thr_s3)
        fold_log.append({"fold": fold, "best_val_auc": best_auc,
                         "thr_g2": thr_g2, "thr_s3": thr_s3})
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # ===== 汇总评估 =====
    results = {"folds": fold_log, "tasks": {}}
    specs = [("g5", "炎症分级 G（5类）", ["G0", "G1", "G2", "G3", "G4"], None),
             ("g2", "明显炎症 G≥2（二分类）", ["<G2", "≥G2"], 1),
             ("s3", "显著纤维化 S≥3（二分类）", ["<S3", "≥S3"], 1)]
    for key, name, names, focus in specs:
        probs = np.concatenate(oof[key])
        labels = np.concatenate(oof_y[key])
        # 二分类任务：用每折自己的验证集门槛（按样本所在折索引）
        thr_arr = None
        if key in oof_thr:
            folds = np.array(oof_fold[key])
            thr_arr = np.array([oof_thr[key][f - 1] for f in folds])
        res, y, pred = evaluate(name, probs, labels, np.array(oof_rep[key]), names, focus,
                                thr_per_sample=thr_arr)
        if key in oof_thr:
            res["thresholds"] = oof_thr[key]
        results["tasks"][key] = res
        plot_cm(name, y, pred, names, os.path.join(OUT_DIR, f"分级V2_{key}_混淆矩阵.png"))
        print(f"\n{name}: Acc={res['acc']:.3f} 宏F1={res['macro_f1']:.3f} AUC={res['auc']:.3f}")
        if focus is not None:
            print(f"  敏感度={res['sensitivity_focus']:.3f} CI={['%.2f'%v for v in res['ci_sens']]}")

    with open(os.path.join(OUT_DIR, "分级V2结果.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n全部完成，结果在 {OUT_DIR}/：分级V2结果.json + 3 张混淆矩阵图")

    # ===== 补丁：保存每个报告的预测标签（供 A5 条件生成）=====
    # 5 折交叉验证下每个报告恰好作为一次"测试折"出现，直接按报告聚合即可
    rep_pred = {}
    for key, thr_key in [("g5", None), ("g2", "g2"), ("s3", "s3")]:
        probs = np.concatenate(oof[key])               # [N, C] 图像级概率
        reps = np.array(oof_rep[key])
        folds = np.array(oof_fold[key])
        uniq, inv = np.unique(reps, return_inverse=True)
        counts = np.bincount(inv).astype(float)
        summed = np.zeros((len(counts), probs.shape[1]))
        np.add.at(summed, inv, probs)
        p = summed / counts[:, None]                   # 报告级平均概率
        for i, rep in enumerate(uniq):
            d = rep_pred.setdefault(rep, {})
            if thr_key is None:
                d["G5"] = int(p[i].argmax())
                d["prob_g5"] = [round(float(v), 4) for v in p[i]]
            else:
                thr = oof_thr[thr_key][folds[np.where(inv == i)[0][0]] - 1]
                d[key.upper()] = int(p[i, 1] >= thr)
                d["prob_" + key] = round(float(p[i, 1]), 4)
    with open(os.path.join(OUT_DIR, "报告预测.json"), "w", encoding="utf-8") as f:
        json.dump(rep_pred, f, ensure_ascii=False, indent=2)
    print(f"已保存报告级预测标签: {len(rep_pred)} 份报告 -> {OUT_DIR}/报告预测.json")


if __name__ == "__main__":
    main()
