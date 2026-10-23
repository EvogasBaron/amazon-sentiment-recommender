"""
Chấm sentiment bằng mô hình pretrained, KHÔNG huấn luyện trên số sao của dự án.

Mô hình mặc định: nlptown/bert-base-multilingual-uncased-sentiment
  - đầu ra 5 lớp tương ứng 1..5 sao (huấn luyện trên review sản phẩm bởi bên thứ ba)
  - điểm = kỳ vọng số sao E = sum(p_k * k), quy về [0, 1]: tf01 = (E - 1) / 4

Văn bản đưa vào mô hình là "tiêu đề tự viết. nội dung". Tiêu đề tự sinh của Amazon
("Five Stars"...) đã được bỏ ở bước prepare_and_eda nên không làm lộ số sao.
Review không có chữ nào (cả tiêu đề lẫn nội dung) nhận tf01 = NaN.

Chạy (từ thư mục gốc dự án):
    pip install torch transformers
    python -m src.sentiment_transformer score            # chấm điểm, có checkpoint, chạy lại được
    python -m src.sentiment_transformer report           # ghép vào sentiment_all + in chẩn đoán

Có GPU thì mất vài phút đến vài chục phút; chỉ CPU thì lâu hơn nhiều, có thể thêm
--max-len 128 để nhanh hơn. Cache nằm theo từng giá trị max-len và từng mô hình, tự xóa nếu văn bản đầu vào đổi.
Đầu ra luôn ghi vào sentiment_transformer.csv.gz (của lần chạy sau cùng).
"""
import argparse
import hashlib

import numpy as np
import pandas as pd

from src.config import CORE_FILE, DATA, SENT_FILE

MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
TF_FILE = DATA / "sentiment_transformer.csv.gz"


def load_docs():
    df = pd.read_csv(CORE_FILE)
    df["text"] = df.text.fillna("")
    df["title"] = df.title.fillna("")
    return df


def make_docs(df):
    """Ghép tiêu đề tự viết và nội dung; bỏ tiền tố '. ' khi không có tiêu đề."""
    return np.where(df.title != "", df.title + ". " + df.text, df.text).tolist()


def score(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    df = load_docs()
    docs = make_docs(df)
    empty = np.array([d.strip() == "" for d in docs])
    order = np.argsort([len(d) for d in docs])      # sắp theo độ dài để batch ít padding

    tag = "" if args.model == MODEL else "_" + args.model.replace("/", "_")   # mô hình khác -> cache riêng
    cache = DATA / f"tf_cache_len{args.max_len}{tag}"
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5("\n".join(docs).encode("utf-8")).hexdigest()
    stamp = cache / "docs.md5"
    if stamp.exists() and stamp.read_text() != digest:
        print("Văn bản đầu vào đã đổi so với lần chạy trước: xóa cache cũ.")
        for f in cache.glob("chunk_*.npy"):
            f.unlink()
    stamp.write_text(digest)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(dev).eval()
    if dev == "cuda":
        model.half()
    print("thiết bị:", dev, "| số review:", len(docs), "| review rỗng:", int(empty.sum()))

    chunk = 8192
    for ci, s in enumerate(range(0, len(order), chunk)):
        f = cache / f"chunk_{ci:04d}.npy"
        if f.exists():                               # chạy lại được sau khi bị ngắt
            continue
        idx = order[s:s + chunk]
        probs = []
        for b in range(0, len(idx), args.batch):
            batch = [docs[i] for i in idx[b:b + args.batch]]
            enc = tok(batch, truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                logits = model(**enc).logits.float()
            probs.append(torch.softmax(logits, -1).cpu().numpy())
        np.save(f, np.concatenate(probs))
        print(f"chunk {ci} xong ({min(s + chunk, len(order))}/{len(order)})", flush=True)

    allp = np.concatenate([np.load(cache / f"chunk_{ci:04d}.npy")
                           for ci in range((len(order) + chunk - 1) // chunk)])
    probs = np.empty_like(allp)
    probs[order] = allp
    exp_star = probs @ np.arange(1, probs.shape[1] + 1, dtype=np.float32)
    exp_star[empty] = np.nan
    out = df[["user_id", "parent_asin"]].copy()
    out["tf_exp_star"] = exp_star
    out["tf01"] = (exp_star - 1) / 4
    out.to_csv(TF_FILE, index=False)
    print("đã lưu", TF_FILE)


def diagnose(d, col, min_words=0):
    """Chẩn đoán điểm sentiment so với số sao. min_words > 0 chỉ xét review đủ dài
    (cần cột n_words); bỏ qua dòng không có điểm."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    d = d.dropna(subset=[col])
    if min_words > 0:
        d = d[d.n_words >= min_words]
    print(f"\n== {col} (review xét: {len(d):,}" + (f", >= {min_words} từ" if min_words else "") + ")")
    g = d.groupby("rating")[col].agg(["mean", "std", "size"]).round(3)
    print(g.to_string())
    means = g["mean"].to_numpy()
    print("đơn điệu tăng theo sao:", bool(np.all(np.diff(means) > 0)))
    sub = d.rating.isin([1, 2, 5])
    if sub.any() and d.rating[sub].nunique() > 1:
        print("AUC (1-2 sao vs 5 sao):", round(roc_auc_score(d.rating[sub] == 5, d[col][sub]), 3))
    neg = d.rating <= 3
    if 0 < neg.sum() < len(d):
        print("AUC (<=3 sao vs >=4 sao):", round(roc_auc_score(~neg, d[col]), 3))
    print("Spearman với sao:", round(spearmanr(d[col], d.rating).statistic, 3))
    low = d[d.rating <= 4]
    print("Spearman chỉ trong nhóm 1-4 sao:", round(spearmanr(low[col], low.rating).statistic, 3))
    five = d[d.rating == 5][col]
    print(f"trong nhóm 5 sao: trung bình {five.mean():.3f}, độ lệch chuẩn {five.std():.3f}")
    print("tỉ lệ bão hòa (>0.9):", round((d[col] > 0.9).mean(), 3))


def report(args):
    df = load_docs()
    df["n_words"] = df.text.str.split().str.len().fillna(0).astype(int)
    tf = pd.read_csv(TF_FILE)
    d = df.merge(tf, on=["user_id", "parent_asin"], validate="1:1")
    diagnose(d, "tf01")
    diagnose(d, "tf01", min_words=15)

    if SENT_FILE.exists():
        allf = pd.read_csv(SENT_FILE).drop(columns=["tf01", "tf_exp_star"], errors="ignore")
    else:
        allf = df[["user_id", "parent_asin"]].copy()
        allf["has_signal"] = (df.title != "") | (df.text != "")
    allf = allf.merge(tf[["user_id", "parent_asin", "tf01"]],
                      on=["user_id", "parent_asin"], validate="1:1")
    allf.to_csv(SENT_FILE, index=False)
    print("\nđã thêm cột tf01 vào", SENT_FILE.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["score", "report"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=256)
    a = ap.parse_args()
    {"score": score, "report": report}[a.stage](a)
