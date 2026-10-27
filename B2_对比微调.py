# -*- coding: utf-8 -*-
"""B2：Chinese-CLIP 域内对比微调（图文检索）
用本院肝脏病理图文对微调，让通用模型学会「病理图像 <-> 病理描述」的对应关系。
损失：双向 InfoNCE（图->文 + 文->图），与 CLIP 原训练一致。
评估：测试集（患者隔离）图->文 / 文->图 Recall@K + 配对 AUC，与零样本对比。
用法：
  HF_HUB_OFFLINE=1 python B2_对比微调.py --data_root . --epochs 50 --batch 32
"""
import os, json, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
from sklearn.metrics import roc_auc_score

HE_TYPES = ["全套餐", "炎症评估", "疑难肝病"]
MODEL_DIRNAME = "chinese_clip模型"


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class PairDataset(Dataset):
    """每张光镜图像与其所属报告的文本构成一对（同报告多图共享文本）"""
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
        else:
            img = img.resize((224, 224))
        pix = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        txt = r["文本"] if pd.notna(r["文本"]) else ""
        enc = self.proc(text=[txt], return_tensors="pt", padding="max_length",
                        truncation=True, max_length=256)
        return (pix, enc["input_ids"][0], enc["attention_mask"][0])


def collate(batch):
    pix = torch.stack([b[0] for b in batch])
    ids = torch.stack([b[1] for b in batch])
    mask = torch.stack([b[2] for b in batch])
    return pix, ids, mask


def embed_all(model, loader, device):
    model.eval()
    FI, FT = [], []
    with torch.no_grad():
        for pix, ids, mask in loader:
            out = model(pixel_values=pix.to(device), input_ids=ids.to(device),
                        attention_mask=mask.to(device))
            fi = out.image_embeds if hasattr(out, "image_embeds") else out[0]
            ft = out.text_embeds if hasattr(out, "text_embeds") else out[1]
            FI.append(F.normalize(fi, dim=1).cpu())
            FT.append(F.normalize(ft, dim=1).cpu())
    return torch.cat(FI).numpy(), torch.cat(FT).numpy()


def retrieval_metrics(FI, FT):
    n = FI.shape[0]
    S = FI @ FT.T
    out = {"配对AUC": round(float(roc_auc_score(np.eye(n).astype(int).ravel(), S.ravel())), 3)}
    for k in (1, 5, 10):
        i2t = (np.argsort(-S, axis=1)[:, :k] == np.arange(n).reshape(-1, 1)).any(1).mean()
        t2i = (np.argsort(-S, axis=0)[:k, :] == np.arange(n).reshape(1, -1)).any(0).mean()
        out[f"Recall@{k}"] = round(float((i2t + t2i) / 2), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default="训练输出_B2")
    args = ap.parse_args()
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    model_path = os.path.join(args.data_root, MODEL_DIRNAME)
    if not os.path.isdir(model_path):
        model_path = "OFA-Sys/chinese-clip-vit-base-patch16"
    print("模型:", model_path, "| device:", device)

    df = pd.read_csv(os.path.join(args.data_root, "建模清单_图像级.csv"))
    df = df[df["报告类型"].isin(HE_TYPES)].reset_index(drop=True)
    df["文本"] = df["诊断意见"].fillna("")
    print(f"光镜图像: {len(df)} 张（训练 {int((df['数据集划分']=='train').sum())} / 测试 {int((df['数据集划分']=='test').sum())}）")

    model = ChineseCLIPModel.from_pretrained(model_path).to(device)
    proc = ChineseCLIPProcessor.from_pretrained(model_path)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    tr = df[df["数据集划分"] == "train"].reset_index(drop=True)
    te = df[df["数据集划分"] == "test"].reset_index(drop=True)
    tr_loader = DataLoader(PairDataset(tr, args.data_root, proc, True), batch_size=args.batch,
                           shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)

    # ---- 微调前先在测试集评估（零样本基线，应在 AUC≈0.5 附近）----
    te_loader = DataLoader(PairDataset(te, args.data_root, proc, False), batch_size=64,
                           shuffle=False, num_workers=4, collate_fn=collate)
    FI0, FT0 = embed_all(model, te_loader, device)
    base = retrieval_metrics(FI0, FT0)
    print("微调前(零样本):", json.dumps(base, ensure_ascii=False))

    # ---- 对比微调 ----
    for ep in range(args.epochs):
        model.train(); tot = n = 0
        for pix, ids, mask in tr_loader:
            out = model(pixel_values=pix.to(device), input_ids=ids.to(device),
                        attention_mask=mask.to(device))
            li = out.logits_per_image; lt = out.logits_per_text
            targets = torch.arange(li.shape[0], device=device)
            loss = (F.cross_entropy(li, targets) + F.cross_entropy(lt, targets)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1}/{args.epochs}  loss={tot/max(n,1):.4f}", flush=True)

    # ---- 微调后评估 ----
    FI1, FT1 = embed_all(model, te_loader, device)
    tuned = retrieval_metrics(FI1, FT1)
    print("微调后:", json.dumps(tuned, ensure_ascii=False))

    result = {"零样本": base, "域内微调后": tuned,
              "配置": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr}}
    with open(os.path.join(args.out, "检索结果.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    np.savez_compressed(os.path.join(args.out, "测试集特征.npz"), image=FI1, text=FT1,
                        研究编号=te["研究编号"].to_numpy())
    torch.save(model.state_dict(), os.path.join(args.out, "检索模型最终.pt"))
    print("已保存到", args.out)


if __name__ == "__main__":
    main()
