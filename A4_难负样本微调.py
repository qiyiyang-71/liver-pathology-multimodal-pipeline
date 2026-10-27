# -*- coding: utf-8 -*-
"""A4：B2 升级版 —— 难负样本挖掘的两阶段对比微调（图文检索）

解决 B2 的失败模式：不同患者的报告共用同一套模板句式，文本高度相似，
模型分不清「这段文字到底配哪张图」（注意：同患者多报告的情况在本数据
的 86 个训练患者中只有 1 例，素材不足以构成难负样本，所以改从
"跨患者的高相似文本"里挖）。

做法（两阶段）：
  阶段1：与 B2 完全相同的信息噪声对比微调（InfoNCE），让模型先学会领域。
  阶段2：用阶段1的模型给全部训练报告文本编码，算出报告间相似度，
         为每份报告找"文本最相似但属于其他报告"的 Top-3 难负样本，
         继续训练并加间隔损失（margin loss）：
           L = mean( relu( cos(img, 难负文本) - cos(img, 正文本) + margin ) )
         强迫模型区分模板化但归属不同的报告。

评估：与 B2 完全同口径（图像级配对AUC、R@1/5/10），
      另加报告级指标（同报告多图特征平均后再算），方便与 B2 的报告级 AUC=0.812 对比。

用法（云端 /root/liver 下）：
  HF_HUB_OFFLINE=1 python A4_难负样本微调.py --data_root . --epochs1 50 --epochs2 20 --batch 32
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
MARGIN = 0.2          # 难负样本间隔
LAMBDA_HARD = 1.0     # 间隔损失权重


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
    """图像级：与 B2 完全相同的口径"""
    n = FI.shape[0]
    S = FI @ FT.T
    out = {"配对AUC": round(float(roc_auc_score(np.eye(n).astype(int).ravel(), S.ravel())), 3)}
    for k in (1, 5, 10):
        i2t = (np.argsort(-S, axis=1)[:, :k] == np.arange(n).reshape(-1, 1)).any(1).mean()
        t2i = (np.argsort(-S, axis=0)[:k, :] == np.arange(n).reshape(1, -1)).any(0).mean()
        out[f"Recall@{k}"] = round(float((i2t + t2i) / 2), 3)
    return out


def report_level_metrics(FI, FT, reps):
    """报告级：同一报告的多张图特征取平均，再算配对 AUC 与 R@K
    （与 B2 的分析口径一致——B2 报告级 AUC=0.812 就是这么做出来的）"""
    reps = np.asarray(reps)
    uniq, inv = np.unique(reps, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    RI = np.zeros((len(uniq), FI.shape[1]))
    np.add.at(RI, inv, FI)
    RI /= counts[:, None]
    RT = FT[np.unique(inv, return_index=True)[1]]   # 同报告文本相同，取首行
    return retrieval_metrics(RI, RT), len(uniq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--epochs1", type=int, default=50, help="阶段1（普通InfoNCE）轮数，与B2一致")
    ap.add_argument("--epochs2", type=int, default=20, help="阶段2（难负样本）轮数")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default="训练输出_A4")
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
    tr = df[df["数据集划分"] == "train"].reset_index(drop=True)
    te = df[df["数据集划分"] == "test"].reset_index(drop=True)
    print(f"光镜图像: {len(df)} 张（训练 {len(tr)} / 测试 {len(te)}）")

    # ---- 构建难负样本表：每份报告 -> 文本最相似的其他报告（跨患者模板化文本）----
    # 阶段1训完后用当前模型编码，再挖难负样本（见 main 中阶段1之后）
    rep_txt = (tr.groupby(["患者组编号", "研究编号"])["文本"].first().reset_index())
    rep_txt["key"] = list(zip(rep_txt["患者组编号"], rep_txt["研究编号"]))
    img_key = list(zip(tr["患者组编号"], tr["研究编号"]))   # 每张图对应的报告 key
    print(f"训练报告数: {len(rep_txt)}（唯一文本 {rep_txt['文本'].nunique()} 份）")

    model = ChineseCLIPModel.from_pretrained(model_path).to(device)
    proc = ChineseCLIPProcessor.from_pretrained(model_path)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    tr_loader = DataLoader(PairDataset(tr, args.data_root, proc, True), batch_size=args.batch,
                           shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
    te_loader = DataLoader(PairDataset(te, args.data_root, proc, False), batch_size=64,
                           shuffle=False, num_workers=4, collate_fn=collate)

    # ---- 微调前测试集评估（零样本基线）----
    FI0, FT0 = embed_all(model, te_loader, device)
    base_img = retrieval_metrics(FI0, FT0)
    base_rep, n_rep = report_level_metrics(FI0, FT0, te["研究编号"].to_numpy())
    print("微调前(零样本) 图像级:", json.dumps(base_img, ensure_ascii=False))
    print("微调前(零样本) 报告级:", json.dumps(base_rep, ensure_ascii=False))

    # ---- 阶段1：普通 InfoNCE 微调（同 B2）----
    for ep in range(args.epochs1):
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
            print(f"[阶段1] epoch {ep+1}/{args.epochs1}  loss={tot/max(n,1):.4f}", flush=True)

    # ---- 阶段2前：用阶段1的模型挖掘文本相似度最高的难负报告 ----
    hard_neg = {}   # 行号 -> 难负文本列表
    if args.epochs2 > 0:
        model.eval()
        texts = rep_txt["文本"].tolist()
        embs = []
        with torch.no_grad():
            for s in range(0, len(texts), 64):
                enc = proc(text=texts[s:s+64], return_tensors="pt", padding=True,
                           truncation=True, max_length=256)
                out = model.text_model(input_ids=enc["input_ids"].to(device),
                                       attention_mask=enc["attention_mask"].to(device))
                pooled = out.pooler_output
                if pooled is None:   # transformers 5.x 不再内置 pooler，取 CLS 位
                    pooled = out.last_hidden_state[:, 0, :]
                e = model.text_projection(pooled)
                embs.append(F.normalize(e, dim=1).cpu())
        embs = torch.cat(embs).numpy()
        sim = embs @ embs.T
        hard_map = {}   # 报告key -> 难负文本Top-3
        for i in range(len(rep_txt)):
            own = texts[i]
            cands = []
            for j in np.argsort(-sim[i]):
                if j == i:
                    continue
                t = texts[int(j)]
                if t and t != own and t not in cands:
                    cands.append(t)
                if len(cands) >= 3:
                    break
            hard_map[rep_txt["key"].iloc[i]] = cands
        n_hard_img = 0
        for i, k in enumerate(img_key):
            if hard_map.get(k):
                hard_neg[i] = hard_map[k]
                n_hard_img += 1
        print(f"有难负样本的图: {n_hard_img}/{len(tr)} 张（Top3相似异报告）")

    # ---- 阶段2：难负样本间隔损失微调 ----
    # 注意：DataLoader shuffle=True 时取不到原行号，所以阶段2自己按随机索引取batch
    if hard_neg and args.epochs2 > 0:
        ds = PairDataset(tr, args.data_root, proc, True)
        n = len(ds)
        bs = args.batch
        for ep in range(args.epochs2):
            model.train(); tot = n_b = 0
            order = np.random.permutation(n)
            for s in range(0, n - bs + 1, bs):
                idx = order[s:s + bs]
                pix, ids, mask = collate([ds[int(j)] for j in idx])
                out = model(pixel_values=pix.to(device), input_ids=ids.to(device),
                            attention_mask=mask.to(device))
                li = out.logits_per_image; lt = out.logits_per_text
                targets = torch.arange(li.shape[0], device=device)
                loss_nce = (F.cross_entropy(li, targets) + F.cross_entropy(lt, targets)) / 2
                # 难负样本间隔损失：逐图编码其难负文本，取该图所有难负中最近的一个
                h_img, h_txt, h_msk = [], [], []
                for row_local, j in enumerate(idx):
                    if int(j) in hard_neg:
                        negs = hard_neg[int(j)]
                        enc = proc(text=[random.choice(negs)], return_tensors="pt",
                                   padding="max_length", truncation=True, max_length=256)
                        h_img.append(row_local)
                        h_txt.append(enc["input_ids"][0])
                        h_msk.append(enc["attention_mask"][0])
                if h_img:
                    htxt = torch.stack(h_txt).to(device)
                    hmask = torch.stack(h_msk).to(device)
                    hout = model.text_model(input_ids=htxt, attention_mask=hmask)
                    pooled = hout.pooler_output
                    if pooled is None:   # transformers 5.x 不再内置 pooler，取 CLS 位
                        pooled = hout.last_hidden_state[:, 0, :]
                    hft = model.text_projection(pooled)
                    hft = F.normalize(hft, dim=1)
                    hfi = F.normalize(out.image_embeds[torch.tensor(h_img, device=device)], dim=1)
                    cos_pos = (hfi * F.normalize(out.text_embeds[torch.tensor(h_img, device=device)], dim=1)).sum(1)
                    cos_neg = (hfi * hft).sum(1)
                    loss_hard = F.relu(cos_neg - cos_pos + MARGIN).mean()
                else:
                    loss_hard = torch.tensor(0.0, device=device)
                loss = loss_nce + LAMBDA_HARD * loss_hard
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); n_b += 1
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"[阶段2] epoch {ep+1}/{args.epochs2}  loss={tot/max(n_b,1):.4f}", flush=True)
    else:
        print("无难负样本或 epochs2=0，跳过阶段2")

    # ---- 微调后评估（图像级 + 报告级，与微调前同口径）----
    FI1, FT1 = embed_all(model, te_loader, device)
    tuned_img = retrieval_metrics(FI1, FT1)
    tuned_rep, _ = report_level_metrics(FI1, FT1, te["研究编号"].to_numpy())
    print("微调后 图像级:", json.dumps(tuned_img, ensure_ascii=False))
    print("微调后 报告级:", json.dumps(tuned_rep, ensure_ascii=False))

    result = {"零样本": {"图像级": base_img, "报告级": base_rep},
              "A4微调后": {"图像级": tuned_img, "报告级": tuned_rep},
              "配置": {"epochs1": args.epochs1, "epochs2": args.epochs2,
                        "batch": args.batch, "lr": args.lr, "margin": MARGIN}}
    with open(os.path.join(args.out, "检索结果.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    np.savez_compressed(os.path.join(args.out, "测试集特征.npz"), image=FI1, text=FT1,
                        研究编号=te["研究编号"].to_numpy())
    torch.save(model.state_dict(), os.path.join(args.out, "检索模型最终.pt"))
    print("已保存到", args.out)


if __name__ == "__main__":
    main()
