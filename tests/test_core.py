"""Kiểm tra nhanh các phần quyết định mọi con số của đồ án.

Chạy từ thư mục gốc dự án:  python tests/test_core.py   (hoặc: python -m pytest tests)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.metrics import evaluate
from src.split import leave_one_out, tie_rate
from src.stats import paired_bootstrap


def test_leave_one_out_and_ties():
    df = pd.DataFrame({
        "user_id":     ["a", "a", "a", "a", "b", "b", "b"],
        "parent_asin": ["i1", "i2", "i3", "i4", "i1", "i2", "i3"],
        "timestamp":   [10, 20, 30, 40, 5, 5, 5],       # user b: cả ba review trùng thời điểm
    })
    tr, va, te = leave_one_out(df.sample(frac=1, random_state=0))   # xáo dòng: kết quả không đổi
    assert te.set_index("user_id").parent_asin.to_dict() == {"a": "i4", "b": "i3"}
    assert va.set_index("user_id").parent_asin.to_dict() == {"a": "i3", "b": "i2"}   # b: phá hòa theo parent_asin
    assert sorted(tr.parent_asin) == ["i1", "i1", "i2"]
    assert tie_rate(df) == 0.5                                      # chỉ user b bị hòa


class _Fixed:
    """Mô hình trả điểm cố định cho từng user."""
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)

    def score(self, users, H):
        return self.scores[users].copy()


def test_evaluate_rank_ties_and_history():
    u2 = {"u0": 0, "u1": 1}
    i2 = {"x0": 0, "x1": 1, "x2": 2, "x3": 3}
    scores = [[0.1, 0.5, 0.5, 0.9],      # u0: item đích x1 hòa với x2, x3 cao hơn
              [0.9, 0.5, 0.2, 0.1]]      # u1: x0 cao nhất nhưng là lịch sử
    H = sp.csr_matrix(([1.0], ([1], [0])), shape=(2, 4))            # u1 đã có x0
    # đưa user theo thứ tự ngược để kiểm tra việc gióng hàng theo user_id
    target = pd.DataFrame({"user_id": ["u1", "u0"], "parent_asin": ["x1", "x1"]})
    out, ranks = evaluate(_Fixed(scores), H, target, u2, i2)
    # hòa xử lý bi quan: x2 bằng điểm nên xếp trước -> u0 có rank 2 (x3 và x2 đứng trước)
    assert ranks.tolist() == [2, 0]                                 # thứ tự u0, u1 (sort theo user_id)
    assert out["HR@5"] == 1.0 and abs(out["NDCG@10"] - (1 / np.log2(4) + 1.0) / 2) < 1e-9


def test_paired_bootstrap_identical_ranks():
    r = np.random.default_rng(0).integers(0, 100, 500)
    d, lo, hi = paired_bootstrap(r, r)
    assert d == lo == hi == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
