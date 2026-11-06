import time
import numpy as np
import pandas as pd
from src.config import SPLITS
from src.models import ItemKNN, ItemPriorRerank, PopFallback, build_matrix
from src.metrics import evaluate

def main():
    t0 = time.time()
    train = pd.read_csv(SPLITS / "train.csv.gz")
    valid = pd.read_csv(SPLITS / "valid.csv.gz")
    test = pd.read_csv(SPLITS / "test.csv.gz")
    allx = pd.concat([train, valid, test])
    user2idx = {u: i for i, u in enumerate(allx.user_id.unique())}
    item2idx = {p: i for i, p in enumerate(allx.parent_asin.unique())}
    X_tr = build_matrix(train, user2idx, item2idx)
    print(f"Loaded matrix in {time.time() - t0:.2f}s")

    print("\n--- 1. ItemKNN Grid Expansion (k) ---")
    for k in (1000, 2000, 3000, 4000, 5000):
        t_fit = time.time()
        m = ItemKNN(k=k, shrink=10.0).fit(X_tr)
        res, _ = evaluate(m, X_tr, valid, user2idx, item2idx)
        print(f"k={k}: NDCG@10={res['NDCG@10']:.5f}, HR@10={res['HR@10']:.5f} ({time.time()-t_fit:.2f}s)")

    print("\n--- 2. E Prior Grid Expansion (lambda) on ItemKNN (k=2000) ---")
    def calc_prior(h, c=10):
        t = pd.DataFrame({"parent_asin": h.parent_asin.to_numpy(), "v": (h.rating - 1) / 4})
        g = t.groupby("parent_asin").v.agg(["sum", "count"])
        sm = (g["sum"] + c * t.v.mean()) / (g["count"] + c)
        z = np.zeros(len(item2idx), dtype=np.float32)
        z[g.index.map(item2idx).to_numpy()] = ((sm - sm.mean()) / sm.std()).to_numpy()
        return z

    z_stars = calc_prior(train)
    base = ItemKNN(2000, 10.0).fit(X_tr)
    # Baseline without prior
    res_base, _ = evaluate(PopFallback(base, X_tr, 1e-4), X_tr, valid, user2idx, item2idx)
    print(f"Baseline (lam=0): NDCG@10={res_base['NDCG@10']:.5f}")

    for lam in (0.005, 0.01, 0.02, 0.05, 0.1, 0.25):
        m = PopFallback(ItemPriorRerank(base, z_stars, lam), X_tr, 1e-4)
        res, _ = evaluate(m, X_tr, valid, user2idx, item2idx)
        print(f"lam={lam}: NDCG@10={res['NDCG@10']:.5f}, HR@10={res['HR@10']:.5f}")

if __name__ == "__main__":
    main()
