# -*- coding: utf-8 -*-
"""基线 V2：在 V1（PLiP 特征 + 逻辑回归）基础上的三项修正
A1 标签修正：纤维化分期为区间（如 S2-3）时取【中点】而非下限（四舍五入，.5 向上）
A2 报告级聚合：一份报告多张图像，对各图预测概率取平均（多实例平均池化）后再出报告级结果
A3 交叉验证：5 折患者级 GroupKFold（全数据 109 名患者），报均值±标准差，替代单次划分
A6 置信区间：报告级指标做患者级 bootstrap（1000 次），给 95% CI
输出：基线V2评估报告.md + 混淆矩阵 PNG（不覆盖 V1 任何文件）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/Users/yangqiyi/Desktop/ai模型/Ai模型"
df = pd.read_csv(os.path.join(ROOT, "建模清单_图像级.csv"))
z = np.load(os.path.join(ROOT, "图像特征_plip.npz"), allow_pickle=True)
feat = pd.DataFrame({"图像路径": z["image_paths"], "_i": range(len(z["image_paths"]))})
df = df.merge(feat, on="图像路径", how="left", validate="1:1")
assert df["_i"].notna().all()
X_all = z["features"][df["_i"].astype(int).to_numpy()]

# ---- A1：标签修正（中点）----
def mid_label(lo, hi):
    if pd.isna(lo):
        return np.nan
    if pd.isna(hi):
        hi = lo
    return int(np.floor((lo + hi) / 2 + 0.5))  # 四舍五入，.5 向上

df["S中点"] = [mid_label(lo, hi) for lo, hi in zip(df["纤维化分期下限"], df["纤维化分期上限"])]
df["G标签"] = df["炎症分级"]
df["split"] = df["数据集划分"]

report_lines = []


def log(s=""):
    print(s)
    report_lines.append(s)


def metrics_at_level(y_true, p_pred, names, level_key):
    """图像级：p_pred 已是逐样本概率，y_true 逐样本标签。
    报告级：按 level_key 分组，对概率取平均、标签取首值（同报告标签一致）。"""
    if level_key is not None:
        _, inv = np.unique(level_key, return_inverse=True)
        n_grp = inv.max() + 1
        counts = np.bincount(inv).astype(np.float64)
        summed = np.zeros((n_grp, p_pred.shape[1]), dtype=np.float64)
        np.add.at(summed, inv, p_pred)
        p_pred = summed / counts[:, None]
        y_true = y_true[np.unique(inv, return_index=True)[1]]
    pred = p_pred.argmax(axis=1)
    acc = accuracy_score(y_true, pred)
    mf1 = f1_score(y_true, pred, average="macro", labels=list(range(len(names))), zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=list(range(len(names))), zero_division=0)
    return acc, mf1, p, r, f1, y_true, pred


def fmt_task(name, names, y_true, p_pred, level_key, ci_unit_keys, cls_binary_focus=None):
    acc, mf1, p, r, f1, yv, pred = metrics_at_level(y_true, p_pred, names, level_key)
    n_img = len(y_true)
    if level_key is not None:
        n_unit = len(yv)
    else:
        n_unit = n_img
    log(f"\n### {name}")
    log(f"- 单位数：{n_unit}（图像 {n_img}）")
    log(f"- 准确率 Accuracy：**{acc:.3f}**")
    log(f"- 宏平均 F1：**{mf1:.3f}**")
    for i, nm in enumerate(names):
        log(f"  - {nm}: {p[i]:.2f} / {r[i]:.2f} / {f1[i]:.2f}")
    if cls_binary_focus is not None:
        log(f"  - 【临床关键】{names[cls_binary_focus]} 召回率（敏感度）：**{r[cls_binary_focus]:.2f}**")
    # 患者级 bootstrap CI（仅报告级）
    if level_key is not None and ci_unit_keys is not None:
        rng = np.random.default_rng(42)
        accs, mf1s, sens = [], [], []
        idx = np.arange(len(yv))
        for _ in range(1000):
            pick = rng.choice(idx, size=len(idx), replace=True)
            if len(np.unique(yv[pick])) < 2:
                continue
            accs.append(accuracy_score(yv[pick], pred[pick]))
            mf1s.append(f1_score(yv[pick], pred[pick], average="macro",
                                 labels=list(range(len(names))), zero_division=0))
            if cls_binary_focus is not None:
                tp = ((pred[pick] == cls_binary_focus) & (yv[pick] == cls_binary_focus)).sum()
                fn = ((pred[pick] != cls_binary_focus) & (yv[pick] == cls_binary_focus)).sum()
                sens.append(tp / (tp + fn) if tp + fn > 0 else np.nan)
        log(f"- 患者级 bootstrap 95% CI：Acc [{np.percentile(accs,2.5):.3f}, {np.percentile(accs,97.5):.3f}]，"
            f"宏F1 [{np.percentile(mf1s,2.5):.3f}, {np.percentile(mf1s,97.5):.3f}]" +
            (f"，敏感度 [{np.nanpercentile(sens,2.5):.3f}, {np.nanpercentile(sens,97.5):.3f}]" if cls_binary_focus is not None else ""))
    return acc, mf1


def plot_cm(name, yv, pred, names, suffix):
    cm = confusion_matrix(yv, pred, labels=list(range(len(names))))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names); ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("预测"); ax.set_ylabel("真实"); ax.set_title(f"{name} 混淆矩阵")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    out = os.path.join(ROOT, f"基线V2_{suffix}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"- 混淆矩阵图：{os.path.basename(out)}")
    return cm


def fit_predict(X_tr, y_tr, X_te, n_cls):
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(sc.transform(X_tr), y_tr)
    return clf.predict_proba(sc.transform(X_te))


TASKS = [
    ("炎症分级 G（5类）", "G标签", ["G0", "G1", "G2", "G3", "G4"], None, None),
    ("纤维化分期 S（4类）", "S中点", ["S1", "S2", "S3", "S4"], None, None),
    ("明显炎症 G≥2（二分类）", "G≥2", ["<G2", "≥G2"], 1, "患者组编号"),
    ("显著纤维化 S≥3（二分类）", "S≥3", ["<S3", "≥S3"], 1, "患者组编号"),
]
df["G≥2"] = (df["G标签"] >= 2).astype(float).where(df["G标签"].notna())
df["S≥3"] = (df["S中点"] >= 3).astype(float).where(df["S中点"].notna())

log("# 基线 V2 评估报告")
log("")
log("特征：PLiP 512 维图像特征（与 V1 相同）；分类器：逻辑回归（class_weight=balanced）。")
log("V2 修正：A1 区间标签取中点（V1 取下限）；A2 报告级=同报告多图预测概率取平均；A3 另做 5 折患者级交叉验证；A6 报告级指标含患者级 bootstrap 95% CI。")

# ========== 第一部分：沿用原划分（train/test），图像级 + 报告级 ==========
log("\n## 一、沿用原患者级划分（测试 21-22 名患者）")
for name, col, names, focus, _ in TASKS:
    mask = df[col].notna()
    sub = df[mask].reset_index(drop=True)
    X = X_all[mask.to_numpy()]
    y = sub[col].astype(int).to_numpy()
    tr = (sub["split"] == "train").to_numpy()
    te = (sub["split"] == "test").to_numpy()
    p_te = fit_predict(X[tr], y[tr], X[te], len(names))
    fmt_task(f"{name} · 图像级", names, y[te], p_te, None, None, focus)
    fmt_task(f"{name} · 报告级", names, y[te], p_te, sub.loc[te, "研究编号"].to_numpy(), focus,
             cls_binary_focus=focus)
    if name.startswith("显著纤维化") or name.startswith("明显炎症"):
        acc, mf1, p, r, f1, yv, pred = metrics_at_level(y[te], p_te, names, sub.loc[te, "研究编号"].to_numpy())
        plot_cm(f"{name} 报告级", yv, pred, names,
                f"{name.split('（')[0].replace(' ','')}_报告级")

# ========== 第二部分：5 折患者级交叉验证（全数据），报告级 ==========
log("\n## 二、5 折患者级交叉验证（全部 109 名患者参与，报告级）")
gkf = GroupKFold(n_splits=5)
for name, col, names, focus, _ in TASKS:
    mask = df[col].notna()
    sub = df[mask].reset_index(drop=True)
    X = X_all[mask.to_numpy()]
    y = sub[col].astype(int).to_numpy()
    groups = sub["患者组编号"].to_numpy()
    fold_acc, fold_mf1, fold_rec = [], [], []
    for tr_idx, te_idx in gkf.split(X, y, groups):
        if len(np.unique(y[te_idx])) < 2:
            continue
        p_te = fit_predict(X[tr_idx], y[tr_idx], X[te_idx], len(names))
        acc, mf1, p, r, f1, yv, pred = metrics_at_level(
            y[te_idx], p_te, names, sub.loc[te_idx, "研究编号"].to_numpy())
        fold_acc.append(acc); fold_mf1.append(mf1)
        if focus is not None:
            fold_rec.append(r[focus])
    log(f"\n### {name} · 报告级 · 5 折 CV")
    log(f"- Accuracy：{np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}（各折 {['%.2f'%v for v in fold_acc]}）")
    log(f"- 宏平均 F1：{np.mean(fold_mf1):.3f} ± {np.std(fold_mf1):.3f}")
    if focus is not None:
        log(f"- 【临床关键】{names[focus]} 召回率：{np.mean(fold_rec):.3f} ± {np.std(fold_rec):.3f}（各折 {['%.2f'%v for v in fold_rec]}）")

out_md = os.path.join(ROOT, "基线V2评估报告.md")
with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
print("\n报告已保存:", out_md)
