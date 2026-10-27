# -*- coding: utf-8 -*-
"""B1：Chinese-CLIP 零样本图文检索评估
图像 <-> 病理文本（诊断意见 / 描述文字），测试集按患者隔离。
"""
import os, time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

ROOT = "/Users/yangqiyi/Desktop/ai模型/Ai模型"
CSV = os.path.join(ROOT, "建模清单_图像级.csv")
MODEL = "OFA-Sys/chinese-clip-vit-base-patch16"
BATCH = 16

df = pd.read_csv(CSV)
# 报告级：每份报告一张代表图（第一张）+ 它的文本，按报告评估避免重复计数
rep = df.drop_duplicates("研究编号").reset_index(drop=True)
te = rep[rep["数据集划分"] == "test"].reset_index(drop=True)
print(f"测试集报告数: {len(te)}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = ChineseCLIPModel.from_pretrained(MODEL).to(device).eval()
proc = ChineseCLIPProcessor.from_pretrained(MODEL)

# ---- 图像特征 ----
t0 = time.time()
imgs = [os.path.join(ROOT, p) for p in te["图像路径"].str.split(";").str[0]]
feats_i = []
def unwrap(f):
    if torch.is_tensor(f):
        return f
    if getattr(f, "pooler_output", None) is not None:
        return f.pooler_output
    if getattr(f, "last_hidden_state", None) is not None:
        return f.last_hidden_state[:, 0]
    raise ValueError(f"无法解析模型输出: {type(f)}")

with torch.no_grad():
    for i in range(0, len(imgs), BATCH):
        im = [Image.open(p).convert("RGB") for p in imgs[i:i + BATCH]]
        x = proc(images=im, return_tensors="pt").to(device)
        f = unwrap(model.get_image_features(**x))
        feats_i.append(f.cpu())
FI = torch.cat(feats_i).numpy()
print(f"图像特征 {FI.shape} 用时 {time.time()-t0:.0f}s")

# ---- 文本特征（两种文本分别评估）----
def text_feats(texts):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            x = proc(text=list(texts[i:i + BATCH]), return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            f = unwrap(model.get_text_features(**x))
            out.append(f.cpu())
    return torch.cat(out).numpy()

def evaluate(FI, FT, name):
    FI_n = FI / np.linalg.norm(FI, axis=1, keepdims=True)
    FT_n = FT / np.linalg.norm(FT, axis=1, keepdims=True)
    S = FI_n @ FT_n.T  # [n_img, n_txt]
    n = S.shape[0]
    # 图->文：每张图的最近文本是否是自己的报告
    i2t = S.argmax(1)
    r1_i = (i2t == np.arange(n)).mean()
    # 文->图
    t2i = S.argmax(0)
    r1_t = (t2i == np.arange(n)).mean()
    # Recall@K（对角线法，双向取平均的常规报告方式）
    def recall_at_k(S, k, axis):
        topk = np.argsort(-S, axis=axis)
        idx = np.arange(n).reshape(-1, 1) if axis == 1 else np.arange(n).reshape(1, -1)
        return (topk[:, :k] == idx).any(axis=1).mean() if axis == 1 else (topk[:k, :] == idx).any(axis=0).mean()
    print(f"\n[{name}] n={n}")
    for k in (1, 5, 10):
        r_i = recall_at_k(S, k, 1)
        r_t = recall_at_k(S, k, 0)
        print(f"  Recall@{k}:  图->文 {r_i:.3f}   文->图 {r_t:.3f}   平均 {(r_i+r_t)/2:.3f}")
    return S

for col in ("诊断意见", "描述文字"):
    texts = te[col].fillna("").tolist()
    FT = text_feats(texts)
    np.savez_compressed(os.path.join(ROOT, f"文本特征_chineseclip_{col}.npz"), features=FT)
    evaluate(FI, FT, col)
