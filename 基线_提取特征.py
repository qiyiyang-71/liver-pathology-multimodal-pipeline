# -*- coding: utf-8 -*-
"""用病理图文预训练模型 PLiP 提取全部图像的特征向量，保存为 npz。
PLiP (Pathology Language-Image Pre-training) 在病理图像-文本对上预训练，
对肝脏病理图像的特征表达远好于通用模型。
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = "/Users/yangqiyi/Desktop/ai模型/Ai模型"
CSV = os.path.join(ROOT, "建模清单_图像级.csv")
OUT = os.path.join(ROOT, "图像特征_plip.npz")
MODEL = "vinid/plip"
BATCH = 16

df = pd.read_csv(CSV)
paths = [os.path.join(ROOT, p) for p in df["图像路径"]]
assert all(os.path.exists(p) for p in paths)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}, 图像数: {len(paths)}")

print("加载模型（首次需下载约 350MB）...")
model = AutoModel.from_pretrained(MODEL).to(device).eval()
proc = AutoProcessor.from_pretrained(MODEL)

feats_list = []
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(paths), BATCH):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH]]
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        out = model.get_image_features(**inputs)  # [B, 512]
        if not torch.is_tensor(out):  # 兼容不同 transformers 版本的返回结构
            if hasattr(out, "image_embeds") and out.image_embeds is not None:
                out = out.image_embeds
            elif out.pooler_output is not None:
                out = out.pooler_output
            else:
                out = out.last_hidden_state[:, 0]
        feats_list.append(out.cpu().numpy().astype(np.float32))
        done = min(i + BATCH, len(paths))
        if done % 160 == 0 or done == len(paths):
            el = time.time() - t0
            print(f"  {done}/{len(paths)}  {el:.0f}s", flush=True)

feats = np.concatenate(feats_list, axis=0)

np.savez_compressed(OUT, features=feats, image_paths=np.array(df["图像路径"]))
print("已保存:", OUT, f"用时 {time.time()-t0:.0f}s")
