# -*- coding: utf-8 -*-
"""B2 报告级聚合分析：图像特征按报告平均后，与报告唯一文本做检索评估"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/Users/yangqiyi/Desktop/ai模型/Ai模型"
df = pd.read_csv(os.path.join(ROOT, "建模清单_图像级.csv"))
rep = df.drop_duplicates("研究编号")
te = rep[rep["数据集划分"] == "test"].reset_index(drop=True)

z = np.load(os.path.join(ROOT, "测试集特征.npz"), allow_pickle=True)
img, txt, reps = z["image"], z["text"], z["研究编号"]
feat_df = pd.DataFrame({"研究编号": reps, "_i": range(len(reps))})
mg = te.merge(feat_df, on="研究编号", how="inner")  # B2 只含光镜报告，电镜/其他不在其中
# 报告级：同一报告多张图像的特征取平均，文本全报告相同取一份
IMG, TXT, rids = [], [], []
for rid, grp in mg.groupby("研究编号"):
    idx = grp["_i"].astype(int).to_numpy()
    IMG.append(img[idx].mean(0))
    TXT.append(txt[idx[0]])
    rids.append(rid)
IMG = np.stack(IMG); TXT = np.stack(TXT)
n = len(rids)
print("报告级评估 n =", n)

S = IMG @ TXT.T
labels = np.eye(n).astype(int)
auc = roc_auc_score(labels.ravel(), S.ravel())

def recall(S, k, axis):
    if axis == 1:
        top = np.argsort(-S, axis=1)[:, :k]
        return (top == np.arange(n).reshape(-1, 1)).any(1).mean()
    top = np.argsort(-S, axis=0)[:k, :]
    return (top == np.arange(n).reshape(1, -1)).any(0).mean()

r = {k: float((recall(S, k, 1) + recall(S, k, 0)) / 2) for k in (1, 5, 10)}
print(f"微调后 报告级: AUC={auc:.3f}  R@1={r[1]:.3f}  R@5={r[5]:.3f}  R@10={r[10]:.3f}")

fpr, tpr, _ = roc_curve(labels.ravel(), S.ravel())
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

axes[0].plot(fpr, tpr, lw=2, label=f"微调后 AUC = {auc:.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray", label="随机 AUC = 0.5")
axes[0].set_xlabel("假阳性率"); axes[0].set_ylabel("真阳性率")
axes[0].set_title("ROC 曲线（报告级，21 名患者）"); axes[0].legend()

metrics = ["配对AUC", "Recall@5", "Recall@10"]
zs = [0.503, 0.044, 0.079]
ft = [auc, r[5], r[10]]
x = np.arange(len(metrics)); w = 0.35
axes[1].bar(x - w/2, zs, w, label="零样本")
axes[1].bar(x + w/2, ft, w, label="域内微调后")
for i, (a, b) in enumerate(zip(zs, ft)):
    axes[1].text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
    axes[1].text(i + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=9)
axes[1].set_xticks(x, metrics); axes[1].set_ylim(0, 1.05)
axes[1].set_title("零样本 vs 域内微调"); axes[1].legend()

im = axes[2].imshow(S, cmap="viridis", vmin=S.min(), vmax=S.max())
axes[2].set_xlabel("文本（按报告）"); axes[2].set_ylabel("图像（按报告）")
axes[2].set_title(f"相似度矩阵（对角线=正确配对）")
for i in range(n):
    axes[2].add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, edgecolor="red", lw=0.8))
fig.colorbar(im, ax=axes[2], shrink=0.8)
fig.tight_layout()
out = os.path.join(ROOT, "B2检索评估图.png")
fig.savefig(out, dpi=160)
print("图已保存:", out)

# 更新检索结果 JSON 加入报告级指标
with open(os.path.join(ROOT, "检索结果.json"), encoding="utf-8") as f:
    res = json.load(f)
res["域内微调后_报告级"] = {"配对AUC": round(float(auc), 3),
                       "Recall@1": round(r[1], 3), "Recall@5": round(r[5], 3),
                       "Recall@10": round(r[10], 3)}
with open(os.path.join(ROOT, "检索结果.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
