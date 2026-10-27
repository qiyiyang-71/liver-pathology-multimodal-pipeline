# -*- coding: utf-8 -*-
"""B0：现代多模态大模型对比实验（本地运行，调 API，无需 GPU）
在和我们完全相同的任务上测评商用大模型：
  grade 任务：对每张光镜图像判断炎症分级 G（0-4）与纤维化分期 S（1-4）
  gen   任务：为测试集报告生成病理描述文本（与 B3 对比 BLEU/ROUGE-L）
支持两种后端（都是 OpenAI 兼容接口）：
  openai    ：OpenAI 官方，模型如 gpt-4o
  dashscope ：阿里云百炼，模型如 qwen-vl-max-latest（即 Qwen2.5-VL）
用法示例：
  pip install openai
  export OPENAI_API_KEY=sk-xxxx
  python B0_大模型分级与生成.py --backend openai --model gpt-4o --task grade --shot 0
  python B0_大模型分级与生成.py --backend dashscope --model qwen-vl-max-latest \
      --dashscope_key sk-yyyy --task grade --shot 5 --limit 60
  python B0_大模型分级与生成.py --backend openai --model gpt-4o --task gen --shot 0
"""
import os, io, json, re, argparse, base64, time
import numpy as np
import pandas as pd
from PIL import Image
from openai import OpenAI
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_fscore_support, confusion_matrix)

ROOT = "/Users/yangqiyi/Desktop/ai模型/Ai模型"
HE_TYPES = ["全套餐", "炎症评估", "疑难肝病"]
OUT_ROOT = os.path.join(ROOT, "B0结果")

GRADE_PROMPT = """你是肝脏病理学专家。观察这张肝穿刺活检病理切片截图（光镜，可能为 HE 或特殊染色），
判断两样东西：
1. 炎症活动度分级 G（0=无，1=轻度，2=轻度至中度，3=中度至重度，4=重度）
2. 纤维化分期 S（1=无或轻度，2=中度，3=显著/桥接纤维化，4=肝硬化）
只回答一行，格式严格如下，不要任何其他文字：
G<数字>,S<数字>"""

GEN_PROMPT = """你是肝脏病理医生。根据这张肝穿刺活检病理切片截图，写出"镜下描述"部分的中文病理描述
（60-150字，专业术语，客观描述所见，不要给出诊断结论，不要编造评分数值）。
直接输出描述文本。"""


def mid_label(lo, hi):
    if pd.isna(lo):
        return np.nan
    if pd.isna(hi):
        hi = lo
    return int(np.floor((lo + hi) / 2 + 0.5))


def img_b64(path, max_side=512):
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def data_url(path):
    return f"data:image/jpeg;base64,{img_b64(path)}"


def make_client(args):
    if args.backend == "openai":
        return OpenAI()  # 读 OPENAI_API_KEY 环境变量
    # dashscope：默认老域名；业务空间密钥可用 --base_url 传专属域名
    base_url = args.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return OpenAI(api_key=args.dashscope_key, base_url=base_url)


def ask(client, model, prompt, image_paths, retries=4):
    """图文提问；image_paths 可多张（5-shot 用）"""
    content = []
    for p in image_paths:
        content.append({"type": "image_url",
                        "image_url": {"url": data_url(p), "detail": "low"}})
    content.append({"type": "text", "text": prompt})
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "user", "content": content}], timeout=120)
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"    重试{attempt+1}: {type(e).__name__} {str(e)[:120]}")
            time.sleep(5 * (attempt + 1))
    return ""


def parse_grade(text):
    m = re.search(r"G\s*[:：]?\s*([0-4])", text or "")
    s = re.search(r"S\s*[:：]?\s*([0-4])", text or "")
    g = int(m.group(1)) if m else None
    sv = int(s.group(1)) if s else None
    if g is None and sv is None:  # 兼容 "G2S3" 连写
        m2 = re.search(r"G\s*([0-4])\s*S\s*([0-4])", text or "")
        if m2:
            g, sv = int(m2.group(1)), int(m2.group(2))
    return g, sv


# ---------- 字符级 BLEU-4（加一平滑）与 ROUGE-L ----------
def char_bleu(pred, ref, n=4):
    from collections import Counter
    def ngrams(s):
        return Counter(s[i:i+k] for k in range(1, n+1) for i in range(len(s)-k+1))
    p, r = ngrams(pred), ngrams(ref)
    if not p or not r:
        return 0.0
    score, tot = 0.0, 0
    for g in p:
        c_pred, c_ref = p[g], r.get(g, 0)
        score += min(c_pred, c_ref + 1) / (c_pred + 1)
        tot += 1
    bp = 1.0 if len(pred) > len(ref) else np.exp(1 - len(ref) / max(1, len(pred)))
    return bp * score / tot


def rouge_l(pred, ref):
    a, b = list(pred), list(ref)
    if not a or not b:
        return 0.0
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[-1][-1]
    p, r = lcs/len(a), lcs/len(b)
    return 2*p*r/max(1e-9, p+r)


def task_grade(args, client, df):
    rng = np.random.default_rng(42)
    out_path = os.path.join(OUT_ROOT, f"grade_{args.backend}_{args.model.replace('/','-')}_shot{args.shot}.jsonl")
    os.makedirs(OUT_ROOT, exist_ok=True)
    todo = df if args.limit <= 0 else df.sample(args.limit, random_state=42)
    if args.shot > 0:  # 从训练患者里挑示例（覆盖不同 G 级）
        exemplars = df[df["数据集划分"] == "train"].groupby("G标签").head(1).head(args.shot)
        shot_imgs = [os.path.join(ROOT, p) for p in exemplars["图像路径"]]
        shot_txt = GRADE_PROMPT + "\n\n参考示例（已给出标准答案，供你学习判断标准）：" + "".join(
            f"\n示例{i+1}：G{int(r['G标签'])},S{int(r['S中点'])}" for i, (_, r) in enumerate(exemplars.iterrows()))
        print(f"5-shot 示例 {len(shot_imgs)} 张")
    records = []
    for i, (_, r) in enumerate(todo.iterrows(), 1):
        imgs = (shot_imgs if args.shot > 0 else []) + [os.path.join(ROOT, r["图像路径"])]
        prompt = shot_txt if args.shot > 0 else GRADE_PROMPT
        text = ask(client, args.model, prompt, imgs)
        g, s = parse_grade(text)
        records.append({"图像路径": r["图像路径"], "研究编号": r["研究编号"], "患者组编号": r["患者组编号"],
                        "G真": None if pd.isna(r["G标签"]) else int(r["G标签"]),
                        "S真": None if pd.isna(r["S中点"]) else int(r["S中点"]),
                        "G预测": g, "S预测": s, "原始回答": text[:300]})
        if i % 20 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("已保存:", out_path)
    return out_path


def score_grade(jsonl_path):
    recs = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
    df = pd.DataFrame(recs)
    df = df.dropna(subset=["G真", "S真"])
    out = {}
    for label_col, pred_col, thr, names, focus in [
            ("G真", "G预测", 2, ["<G2", "≥G2"], 1),
            ("S真", "S预测", 3, ["<S3", "≥S3"], 1)]:
        df[f"真二值"] = (df[label_col] >= thr).astype(int)
        df[f"预二值"] = (df[pred_col] >= thr).astype(float)
        # 报告级：分数=报告内图像判为阳性的比例，标签=首值
        grp = df.groupby("研究编号")
        score = grp["预二值"].mean()
        y = grp[f"真二值"].first()
        m = y.notna() & score.notna()
        y, score = y[m].astype(int), score[m].astype(float)
        pred = (score >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        sens = ((pred == 1) & (y == 1)).sum() / max(1, y.sum())
        spec = ((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())
        auc = roc_auc_score(y, score) if len(np.unique(y)) == 2 else float("nan")
        out[names[1]] = {"n_reports": int(len(y)), "acc": round(acc, 3),
                         "sensitivity": round(sens, 3), "specificity": round(spec, 3),
                         "auc": round(auc, 3)}
        cm = confusion_matrix(y, pred)
        out[names[1]]["confusion"] = cm.tolist()
    out_path = jsonl_path.replace(".jsonl", "_评分.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("评分已保存:", out_path)


def task_gen(args, client, df):
    test = df[df["数据集划分"] == "test"]
    reps = test.groupby("研究编号").first().reset_index()
    if args.limit > 0:
        reps = reps.head(args.limit)
    out_path = os.path.join(OUT_ROOT, f"gen_{args.backend}_{args.model.replace('/','-')}_shot{args.shot}.csv")
    rows = []
    for i, (_, r) in enumerate(reps.iterrows(), 1):
        imgs = [os.path.join(ROOT, p) for p in test[test["研究编号"] == r["研究编号"]]["图像路径"].tolist()[:4]]
        text = ask(client, args.model, GEN_PROMPT, imgs)
        ref = r["描述文字"] if pd.notna(r["描述文字"]) else ""
        rows.append({"研究编号": r["研究编号"], "生成": text, "参考": ref,
                     "BLEU": round(char_bleu(text, ref), 3), "ROUGE_L": round(rouge_l(text, ref), 3)})
        print(f"  {i}/{len(reps)} BLEU={rows[-1]['BLEU']:.2f} ROUGE-L={rows[-1]['ROUGE_L']:.2f}")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n均值: BLEU={np.mean([r['BLEU'] for r in rows]):.3f}  "
          f"ROUGE-L={np.mean([r['ROUGE_L'] for r in rows]):.3f}  (B3 模型: 0.260 / 0.456)")
    print("已保存:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai", "dashscope"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dashscope_key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    ap.add_argument("--task", choices=["grade", "gen"], required=True)
    ap.add_argument("--shot", type=int, default=0, help="0=零样本 5=5-shot")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用），0=全部")
    ap.add_argument("--score_only", default="", help="只对已有的 grade jsonl 评分")
    ap.add_argument("--base_url", default="",
                    help="dashscope 业务空间专属 Base URL（API 页面上那个 https://ws-... 地址）")
    args = ap.parse_args()
    if args.score_only:
        return score_grade(args.score_only)
    if args.model is None:
        args.model = "gpt-4o" if args.backend == "openai" else "qwen-vl-max-latest"

    df = pd.read_csv(os.path.join(ROOT, "建模清单_图像级.csv"))
    df = df[df["报告类型"].isin(HE_TYPES)].reset_index(drop=True)
    df["S中点"] = [mid_label(lo, hi) for lo, hi in zip(df["纤维化分期下限"], df["纤维化分期上限"])]
    df["G标签"] = df["炎症分级"]

    client = make_client(args)
    if args.task == "grade":
        path = task_grade(args, client, df)
        print("\n===== 报告级评分 =====")
        score_grade(path)
    else:
        task_gen(args, client, df)


if __name__ == "__main__":
    main()
