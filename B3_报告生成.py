# -*- coding: utf-8 -*-
"""B3：病理图像自动描述生成（ClipCap 架构）
视觉编码：Chinese-CLIP ViT-B/16（冻结；若存在 B2 微调权重则优先加载）
文本解码：GPT-2 中文（uer/gpt2-distil-chinese-cluecorpussmall，本地文件夹 gpt2模型/）
训练：前缀映射网络 + 解码器，教师强制；评估：BLEU/ROUGE-L + 生成样例
用法：
  HF_HUB_OFFLINE=1 python B3_报告生成.py --data_root . --epochs 30 --batch 8
"""
import os, json, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (ChineseCLIPModel, ChineseCLIPProcessor,
                          GPT2LMHeadModel, BertTokenizer,
                          get_cosine_schedule_with_warmup)
from collections import Counter

HE_TYPES = ["全套餐", "炎症评估", "疑难肝病"]
CLIP_DIRNAME = "chinese_clip模型"
GPT2_DIRNAME = "gpt2模型"
PREFIX_LEN = 16
MAX_TEXT_LEN = 320


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class CaptionDataset(Dataset):
    def __init__(self, df, data_root, clip_proc, tok, train=True):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.clip_proc = clip_proc
        self.tok = tok
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
        pix = self.clip_proc(images=img, return_tensors="pt")["pixel_values"][0]
        ids = self.tok(r["文本"], max_length=MAX_TEXT_LEN, truncation=True)["input_ids"]
        return pix, torch.tensor(ids, dtype=torch.long)


class CaptionModel(nn.Module):
    def __init__(self, clip_model, gpt2, n_embd):
        super().__init__()
        self.vision = clip_model.vision_model          # 冻结
        for p in self.vision.parameters():
            p.requires_grad = False
        self.map_net = nn.Sequential(
            nn.Linear(768, 768), nn.GELU(),
            nn.Linear(768, PREFIX_LEN * n_embd), nn.GELU(),
        )
        self.gpt2 = gpt2
        self.prefix_len = PREFIX_LEN

    def forward(self, pix, ids):
        device = pix.device
        with torch.no_grad():
            vfeat = self.vision(pixel_values=pix).pooler_output      # [B,768]
        prefix = self.map_net(vfeat).view(-1, self.prefix_len, self.gpt2.config.n_embd)
        tok_emb = self.gpt2.transformer.wte(ids)                    # [B,T,E]
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)         # [B,P+T,E]
        attn = torch.cat([torch.ones(ids.shape[0], self.prefix_len, dtype=torch.long, device=device),
                          (ids != self.tok_pad).long()], dim=1)
        labels = torch.cat([torch.full((ids.shape[0], self.prefix_len), -100, dtype=torch.long, device=device),
                            ids.clone()], dim=1)
        labels[:, self.prefix_len:][ids == self.tok_pad] = -100
        return self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)

    tok_pad = 0  # 运行前由 main 设置


def build_bert_tokenizer(path):
    tok = BertTokenizer.from_pretrained(path)
    if "[SEP]" in tok.get_vocab():
        tok.eos_token = "[SEP]"
    if tok.pad_token is None:
        tok.pad_token = "[PAD]"
    return tok


# ---------- 字符级 BLEU & ROUGE-L ----------
def ngrams(chars, n):
    return Counter(tuple(chars[i:i+n]) for i in range(len(chars) - n + 1))

def bleu_char(pred, ref, max_n=4):
    if not pred or not ref:
        return 0.0
    bp = 1.0 if len(pred) > len(ref) else np.exp(1 - len(ref) / max(len(pred), 1))
    logp = 0.0
    for n in range(1, max_n + 1):
        pn, dn = ngrams(list(pred), n), ngrams(list(ref), n)
        if not pn:
            continue
        overlap = sum((pn & dn).values())
        logp += np.log((overlap + 1) / (sum(pn.values()) + 1))  # 加一平滑
    return float(bp * np.exp(logp / max_n))

def lcs_len(a, b):
    m, n = len(a), len(b)
    dp = [0] * (n + 1)  # 必须全零初始化
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if a[i-1] == b[j-1] else max(dp[j], dp[j-1])
            prev = tmp
    return dp[n]

def rouge_l(pred, ref):
    if not pred or not ref:
        return 0.0
    l = lcs_len(list(pred), list(ref))
    p, r = l / len(pred), l / len(ref)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--text_col", default="诊断意见")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--b2_ckpt", default="训练输出_B2/检索模型最终.pt")
    ap.add_argument("--out", default="训练输出_B3")
    args = ap.parse_args()
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    clip_path = os.path.join(args.data_root, CLIP_DIRNAME)
    if not os.path.isdir(clip_path):
        clip_path = "OFA-Sys/chinese-clip-vit-base-patch16"
    gpt2_path = os.path.join(args.data_root, GPT2_DIRNAME)
    print("clip:", clip_path, "| gpt2:", gpt2_path, "| device:", device)

    clip = ChineseCLIPModel.from_pretrained(clip_path).to(device).eval()
    # 若 B2 微调权重存在，加载其视觉部分（对齐更好）
    b2 = os.path.join(args.data_root, args.b2_ckpt)
    if os.path.exists(b2):
        sd = torch.load(b2, map_location="cpu")
        missing, unexpected = clip.load_state_dict(sd, strict=False)
        print(f"已加载 B2 微调权重: {b2} (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("未找到 B2 权重，使用基础 Chinese-CLIP")
    clip_proc = ChineseCLIPProcessor.from_pretrained(clip_path)
    gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_path).to(device)
    tok = build_bert_tokenizer(gpt2_path)
    print("pad_token:", tok.pad_token, "| eos_token:", tok.eos_token)

    model = CaptionModel(clip, gpt2, gpt2.config.n_embd).to(device)
    model.tok_pad = tok.pad_token_id
    CaptionModel.tok_pad = tok.pad_token_id

    df = pd.read_csv(os.path.join(args.data_root, "建模清单_图像级.csv"))
    df = df[df["报告类型"].isin(HE_TYPES)].reset_index(drop=True)
    df["文本"] = df[args.text_col].fillna("").str.strip()
    df = df[df["文本"].str.len() >= 5].reset_index(drop=True)
    tr = df[df["数据集划分"] == "train"].reset_index(drop=True)
    te = df[df["数据集划分"] == "test"].reset_index(drop=True)
    print(f"图文对: 训练 {len(tr)} / 测试 {len(te)}；目标文本列: {args.text_col}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(tr) // args.batch + 1)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    def collate(batch):
        pix = torch.stack([b[0] for b in batch])
        ids = [b[1] for b in batch]
        maxlen = max(len(x) for x in ids)
        ids = torch.stack([torch.cat([x, torch.full((maxlen - len(x),), tok.pad_token_id, dtype=torch.long)])
                           if len(x) < maxlen else x for x in ids])
        return pix, ids

    loader = DataLoader(CaptionDataset(tr, args.data_root, clip_proc, tok, True),
                        batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                        collate_fn=collate)

    for ep in range(args.epochs):
        model.train(); tot = n = 0
        for pix, ids in loader:
            pix, ids = pix.to(device), ids.to(device)
            out = model(pix, ids)
            loss = out.loss
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); n += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1}/{args.epochs}  loss={tot/max(n,1):.4f}", flush=True)

    # ---- 测试集生成 ----
    print("训练完成，开始在测试集生成...")
    model.eval()
    rep_te = te.drop_duplicates("研究编号").reset_index(drop=True)
    rows, bleu_s, rouge_s = [], [], []
    for _, r in rep_te.iterrows():
        img = Image.open(os.path.join(args.data_root, r["图像路径"])).convert("RGB").resize((224, 224))
        pix = clip_proc(images=img, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            vfeat = model.vision(pixel_values=pix).pooler_output
            prefix = model.map_net(vfeat).view(1, PREFIX_LEN, model.gpt2.config.n_embd)
        attn = torch.ones(1, PREFIX_LEN, dtype=torch.long, device=device)
        with torch.no_grad():
            gen = model.gpt2.generate(inputs_embeds=prefix, attention_mask=attn,
                                      max_new_tokens=MAX_TEXT_LEN, num_beams=4,
                                      early_stopping=True, pad_token_id=tok.pad_token_id,
                                      eos_token_id=tok.eos_token_id)
        pred = tok.decode(gen[0], skip_special_tokens=True).replace(" ", "").strip()
        ref = r["文本"]
        b, rg = bleu_char(pred, ref), rouge_l(pred, ref)
        bleu_s.append(b); rouge_s.append(rg)
        rows.append({"研究编号": r["研究编号"], "生成描述": pred, "真实描述": ref,
                     "BLEU": round(b, 4), "ROUGE-L": round(rg, 4)})

    res = {"测试报告数": len(rows),
           "BLEU_字符级": round(float(np.mean(bleu_s)), 4),
           "ROUGE-L": round(float(np.mean(rouge_s)), 4),
           "配置": {"text_col": args.text_col, "epochs": args.epochs, "batch": args.batch, "lr": args.lr}}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open(os.path.join(args.out, "生成结果.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "测试集生成明细.csv"),
                              index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), os.path.join(args.out, "生成模型最终.pt"))
    print("已保存到", args.out)


if __name__ == "__main__":
    main()
