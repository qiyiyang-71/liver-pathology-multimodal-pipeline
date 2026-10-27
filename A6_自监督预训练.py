# -*- coding: utf-8 -*-
"""A6：域内自监督继续预训练（SimCLR，云端 GPU）
动机: PLiP 预训练于英文社交媒体病理图，与本院中文肝病穿刺切片存在域差；
     用 WSI 无标签切块（上万张，含全部染色）做 SimCLR 继续预训练视觉编码器，
     让模型先学会本领域的组织形态表征，再接 V3 监督微调（=V4）。
用法:
  HF_ENDPOINT=https://hf-mirror.com python A6_自监督预训练.py --data_root . --epochs 50 --batch 256
  data_root 下需有: 图文数据集_SSL/（无标签切块池）+ plip模型/（或走镜像下载）
产物: plip模型_SSL/（可直接作为 V3 微调脚本的 --model 输入，即为 V4）
预计: RTX 4090, batch 256, 50 epochs 约 1.5–2.5 小时
"""
import os, argparse, random, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoProcessor
from torchvision import transforms

MODEL_NAME = "vinid/plip"
OUT_MODEL = "plip模型_SSL"


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class SSLPool(Dataset):
    """无标签切块池：同一块做两次独立增广 = SimCLR 正样本对"""

    def __init__(self, root, proc):
        self.paths = sorted(glob.glob(os.path.join(root, "图文数据集_SSL", "*", "*.jpg")))
        assert self.paths, "未找到 图文数据集_SSL/*/*.jpg"
        self.proc = proc
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.15, hue=0.03)], p=0.8),
            transforms.RandomGrayscale(p=0.05),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        v1 = self.proc(images=self.aug(img), return_tensors="pt")["pixel_values"][0]
        v2 = self.proc(images=self.aug(img), return_tensors="pt")["pixel_values"][0]
        return v1, v2


class SimCLRNet(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_path).vision_model
        hid = self.enc.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 128))

    def forward(self, x):
        emb = self.enc(pixel_values=x).pooler_output
        return F.normalize(self.proj(emb), dim=-1)


def nt_xent(z1, z2, tau=0.5):
    B = z1.shape[0]
    z = torch.cat([z1, z2])                       # [2B, d]
    sim = z @ z.t() / tau                         # [2B, 2B]
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)             # i 的配对是 i±B
    return F.cross_entropy(sim, targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--model", default=None, help="默认 <data_root>/plip模型 或 vinid/plip")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=OUT_MODEL)
    args = ap.parse_args()
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = args.model or (
        os.path.join(args.data_root, "plip模型")
        if os.path.isdir(os.path.join(args.data_root, "plip模型")) else MODEL_NAME)
    print(f"基座: {model_path}")
    proc = AutoProcessor.from_pretrained(model_path)

    ds = SSLPool(args.data_root, proc)
    print(f"SSL 图像池: {len(ds)} 块")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True)

    net = SimCLRNet(model_path).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, len(dl) * args.epochs))

    for ep in range(1, args.epochs + 1):
        net.train()
        tot, n = 0.0, 0
        for v1, v2 in dl:
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
            loss = nt_xent(net(v1), net(v2))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); n += 1
        print(f"epoch {ep}/{args.epochs} loss={tot/n:.4f}", flush=True)

    # 把训练好的视觉编码器塞回完整 PLiP 模型再保存，V3 脚本可直接加载
    full = AutoModel.from_pretrained(model_path)
    full.vision_model.load_state_dict(net.enc.state_dict())
    full.save_pretrained(args.out)
    proc.save_pretrained(args.out)
    print(f"完成 -> {args.out}/（V4 微调: python 分级V3_微调PLiP.py --model {args.out}）")


if __name__ == "__main__":
    main()
