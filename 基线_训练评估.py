# -*- coding: utf-8 -*-
"""基线模型：PLiP 图像特征 + 逻辑回归，预测炎症分级(G)与纤维化分期(S)。
评估在患者级划分的测试集上进行（22 名患者，训练/测试无患者重叠）。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as as_plt
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

df["split"] = df["数据集划分"]
report_lines = []


def log(s=""):
    print(s)
    report_lines.append(s)


def run_task(name, y_series, class_names=None, binary=False):
    """y_series: 与 df 对齐的标签（含 NaN，NaN 行剔除）"""
    mask = y_series.notna()
    sub = df[mask].reset_index(drop=True)
    X, y = X_all[mask.to_numpy()], y_series[mask].astype(int).to_numpy()
    tr = (sub["split"] == "train").to_numpy()
    te = (sub["split"] == "test").to_numpy()
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(sc.transform(X[tr]), y[tr])
    pred = clf.predict(sc.transform(X[te]))
    acc = accuracy_score(y[te], pred)
    mf1 = f1_score(y[te], pred, average="macro")
    labels = np.unique(np.concatenate([y, pred]))
    p, r, f1, _ = precision_recall_fscore_support(y[te], pred, labels=labels, zero_division=0)
    # 多数类基准
    maj = np.bincount(y[tr]).argmax()
    maj_acc = (y[te] == maj).mean()
    log(f"\n### {name}（测试集 {te.sum()} 张图像 / {sub[te]['研究编号'].nunique()} 份报告 / {sub[te]['患者组编号'].nunique()} 名患者）")
    log(f"- 准确率 Accuracy：**{acc:.3f}**（多数类基准 {maj_acc:.3f}）")
    log(f"- 宏平均 F1：**{mf1:.3f}**")
    names = class_names if class_names else [str(l) for l in labels]
    log("- 各类别表现（precision / recall / F1）:")
    for l, pi, ri, fi in zip(labels, p, r, f1):
        nm = names[l] if binary is False and l < len(names) else str(l)
        log(f"  - {nm}: {pi:.2f} / {ri:.2f} / {fi:.2f}")
    # 混淆矩阵图
    order = np.arange(len(names))
    cm = confusion_matrix(y[te], pred, labels=order)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(order, names); ax.set_yticks(order, names)
    ax.set_xlabel("预测"); ax.set_ylabel("真实"); ax.set_title(f"{name} 混淆矩阵")
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    out = os.path.join(ROOT, f"基线结果_{name.replace(' ', '_')}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"- 混淆矩阵图：{os.path.basename(out)}")
    return acc, mf1


log("# 基线模型评估报告")
log("")
log("特征：PLiP 病理图文预训练模型（vinid/plip）提取的 512 维图像特征；")
log("分类器：逻辑回归（类别不平衡加权）；测试集按患者隔离（22 名患者）。")
log("说明：纤维化分期为区间（如 2-3）时取下限作为标签；无标签行已剔除。")

run_task("炎症分级 G（5 类）", df["炎症分级"],
         class_names=["G0", "G1", "G2", "G3", "G4"])
run_task("纤维化分期 S（4 类）", df["纤维化分期下限"],
         class_names=["S1", "S2", "S3", "S4"])
run_task("明显炎症 G≥2（二分类）", (df["炎症分级"] >= 2).astype(float).where(df["炎症分级"].notna()),
         class_names=["<G2", "≥G2"])
run_task("显著纤维化 S≥3（二分类）", (df["纤维化分期下限"] >= 3).astype(float).where(df["纤维化分期下限"].notna()),
         class_names=["<S3", "≥S3"])

out_md = os.path.join(ROOT, "基线评估报告.md")
with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
print("\n报告已保存:", out_md)
