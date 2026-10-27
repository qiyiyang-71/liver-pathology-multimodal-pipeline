# -*- coding: utf-8 -*-
"""A5：B3 升级版 —— 结构化条件约束的病理报告生成

与 B3 的区别：生成时把「炎症分级 G / 纤维化分期 S」作为结构化条件
以文字前缀的形式喂给 GPT-2 解码器，让报告内容受病理分级约束，
避免 B3 出现的「生成流畅但与真实分级不符」的问题。

关键设计：条件标签用【预测标签】而不是真值标签——
  真实使用管线是：图像 -> 分级模型(分级V2) -> 预测G/S -> 生成模型 -> 报告
  训练、测试都走这条管线，指标才反映真实可用性。
  预测标签来自 训练输出_分级V2/报告预测.json（先跑 分级V2_微调PLiP_保存预测.py）。

架构与 B3 相同：Chinese-CLIP ViT（冻结）+ 16-token 前缀映射 + GPT-2 中文解码。
训练：教师强制，输入 = [条件前缀文字] + [真实报告文本]；
生成：输入 = [预测条件前缀] + [图像前缀]，beam search 解码。

用法（云端 /root/liver 下，按顺序）：
  1) HF_ENDPOINT=https://hf-mirror.com python 分级V2_微调PLiP_保存预测.py --data_root . --epochs 12 --batch 32
  2) HF_HUB_OFFLINE=1 python A5_条件生成.py --data_root . --epochs 30 --batch 8
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
MAX_TEXT_LEN = 320      # 报告文本上限（与 B3 相同）
MAX_COND_LEN = 48       # 条件前缀文字上限（足够容纳 G/S 描述）


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_condition(g5, s3, mode="g5"):
    """把预测标签转成自然语言前缀（训练与生成必须用同一个函数）
    mode="g5"   : 五分类G + S3二分类（粒度细但噪声大）
    mode="binary": G≥2/S≥3 二分类（更可靠，分级模型敏感度0.91/0.76）"""
    s3 = int(s3)
    if mode == "binary":
        g2 = int(g5)
        g_txt = "存在明显炎症（G≥2）" if g2 == 1 else "炎症较轻（G<2）"
        s_txt = "显著纤维化（S≥3）" if s3 == 1 else "纤维化较轻（S<3）"
        return f"该患者{g_txt}，{s_txt}。病理描述如下："
    g5 = int(g5)
    s_txt = "纤维化分期S3及以上" if s3 == 1 else "纤维化分期S3以下"
    return f"该患者炎症分级G{g5}，{s_txt}。病理描述如下："


class CaptionDataset(Dataset):
    def __init__(self, df, data_root, clip_proc, tok, cond_map, train=True):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.clip_proc = clip_proc
        self.tok = tok
        self.cond_map = cond_map    # 研究编号 -> 条件文字
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
        cond = self.cond_map.get(r["研究编号"], "")
        full = cond + (r["文本"] if pd.notna(r["文本"]) else "")
        ids = self.tok(full, max_length=MAX_TEXT_LEN + MAX_COND_LEN, truncation=True)["input_ids"]
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


# ---------- 字符级 BLEU & ROUGE-L（与 B3 完全相同的实现）----------
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
    dp = [0] * (n + 1)
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


def collate(batch, pad_id):
    pix = torch.stack([b[0] for b in batch])
    ids = [b[1] for b in batch]
    maxlen = max(len(x) for x in ids)
    ids = torch.stack([torch.cat([x, torch.full((maxlen - len(x),), pad_id, dtype=torch.long)])
                       if len(x) < maxlen else x for x in ids])
    return pix, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--text_col", default="诊断意见")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--b2_ckpt", default="训练输出_B2/检索模型最终.pt")
    ap.add_argument("--pred_json", default="训练输出_分级V2/报告预测.json")
    ap.add_argument("--label_source", choices=["pred", "oracle"], default="pred",
                    help="pred=分级模型预测标签(真实管线)；oracle=真值标签(上限对照)")
    ap.add_argument("--cond_mode", choices=["g5", "binary"], default="g5",
                    help="g5=五分类G条件；binary=G≥2/S≥3二分类条件(更可靠)")
    ap.add_argument("--out", default="训练输出_A5")
    args = ap.parse_args()
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    clip_path = os.path.join(args.data_root, CLIP_DIRNAME)
    if not os.path.isdir(clip_path):
        clip_path = "OFA-Sys/chinese-clip-vit-base-patch16"
    gpt2_path = os.path.join(args.data_root, GPT2_DIRNAME)
    print("clip:", clip_path, "| gpt2:", gpt2_path, "| device:", device)

    # ---- 条件标签来源：预测标签（真实管线）或真值标签（上限对照）----
    cond_map = {}
    if args.label_source == "pred":
        pred_path = os.path.join(args.data_root, args.pred_json)
        with open(pred_path, encoding="utf-8") as f:
            rep_pred = json.load(f)
        if args.cond_mode == "binary":
            cond_map = {rep: build_condition(d["G2"], d["S3"], "binary") for rep, d in rep_pred.items()}
        else:
            cond_map = {rep: build_condition(d["G5"], d["S3"], "g5") for rep, d in rep_pred.items()}
        print(f"已加载 {len(cond_map)} 份报告的预测标签（{args.pred_json}，模式 {args.cond_mode}）")
    else:
        print("使用真值标签作为条件（oracle 上限对照）")

    clip = ChineseCLIPModel.from_pretrained(clip_path).to(device).eval()
    b2 = os.path.join(args.data_root, args.b2_ckpt)
    if os.path.exists(b2):
        sd = torch.load(b2, map_location="cpu")
        missing, unexpected = clip.load_state_dict(sd, strict=False)
        print(f"已加载检索微调权重: {b2} (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("未找到检索微调权重，使用基础 Chinese-CLIP")
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

    if args.label_source == "oracle":
        # 真值标签：G 取炎症分级，S 取区间中点（与分级V2同一套标签规则）
        def mid_label(lo, hi):
            if pd.isna(lo):
                return np.nan
            if pd.isna(hi):
                hi = lo
            return int(np.floor((lo + hi) / 2 + 0.5))
        rep_true = df.groupby("研究编号").first().reset_index()
        for _, r in rep_true.iterrows():
            if pd.isna(r["炎症分级"]):
                continue
            g5 = int(r["炎症分级"])
            s_mid = mid_label(r["纤维化分期下限"], r["纤维化分期上限"])
            s3 = 0 if pd.isna(s_mid) else int(s_mid >= 3)
            if args.cond_mode == "binary":
                cond_map[r["研究编号"]] = build_condition(int(g5 >= 2), s3, "binary")
            else:
                cond_map[r["研究编号"]] = build_condition(g5, s3, "g5")
        print(f"已加载 {len(cond_map)} 份报告的真值标签")
    print("条件示例:", next(iter(cond_map.values()), "（无）"))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(tr) // args.batch + 1)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    from functools import partial
    loader = DataLoader(CaptionDataset(tr, args.data_root, clip_proc, tok, cond_map, True),
                        batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True,
                        collate_fn=partial(collate, pad_id=tok.pad_token_id))

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

    # ---- 测试集生成：条件前缀用分级模型的【预测】标签，模拟真实管线 ----
    print(f"训练完成，开始在测试集生成（条件来源: {args.label_source}）...")
    model.eval()
    rep_te = te.drop_duplicates("研究编号").reset_index(drop=True)
    rows, bleu_s, rouge_s = [], [], []
    for _, r in rep_te.iterrows():
        img = Image.open(os.path.join(args.data_root, r["图像路径"])).convert("RGB").resize((224, 224))
        pix = clip_proc(images=img, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            vfeat = model.vision(pixel_values=pix).pooler_output
            prefix = model.map_net(vfeat).view(1, PREFIX_LEN, model.gpt2.config.n_embd)
        # 图像前缀 + 条件文字前缀（GPT-2 embedding 空间）
        cond = cond_map.get(r["研究编号"], "")
        cond_ids = tok(cond, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            cond_emb = model.gpt2.transformer.wte(cond_ids)
        inputs_embeds = torch.cat([prefix, cond_emb], dim=1)
        attn = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long, device=device)
        with torch.no_grad():
            gen = model.gpt2.generate(inputs_embeds=inputs_embeds, attention_mask=attn,
                                      max_new_tokens=MAX_TEXT_LEN, num_beams=4,
                                      early_stopping=True, pad_token_id=tok.pad_token_id,
                                      eos_token_id=tok.eos_token_id)
        pred_txt = tok.decode(gen[0], skip_special_tokens=True).replace(" ", "").strip()
        # 去掉可能复述的条件前缀，只保留生成的报告正文
        for marker in ["病理描述如下：", "病理描述如下:"]:
            if marker in pred_txt:
                pred_txt = pred_txt.split(marker, 1)[1]
                break
        ref = r["文本"]
        b, rg = bleu_char(pred_txt, ref), rouge_l(pred_txt, ref)
        bleu_s.append(b); rouge_s.append(rg)
        rows.append({"研究编号": r["研究编号"], "条件": cond, "生成描述": pred_txt,
                     "真实描述": ref, "BLEU": round(b, 4), "ROUGE-L": round(rg, 4)})

    res = {"测试报告数": len(rows),
           "BLEU_字符级": round(float(np.mean(bleu_s)), 4),
           "ROUGE-L": round(float(np.mean(rouge_s)), 4),
           "配置": {"text_col": args.text_col, "epochs": args.epochs, "batch": args.batch,
                     "lr": args.lr, "label_source": args.label_source, "cond_mode": args.cond_mode}}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open(os.path.join(args.out, "生成结果.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "测试集生成明细.csv"),
                              index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), os.path.join(args.out, "生成模型最终.pt"))
    print("已保存到", args.out)


if __name__ == "__main__":
    main()
