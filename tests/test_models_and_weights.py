"""
Unit tests kiểm tra toàn diện các mô hình gợi ý và hàm xử lý trọng số:
- Popularity, ItemKNN, EASE
- PopFallback, ItemPriorRerank
- build_matrix, build_weighted
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.models import Popularity, ItemKNN, EASE, PopFallback, ItemPriorRerank, build_matrix, build_weighted


def test_build_matrix_and_weighted():
    df = pd.DataFrame({
        "user_id": ["u0", "u0", "u1", "u2"],
        "parent_asin": ["i0", "i1", "i1", "i2"],
        "weight": [1.0, 0.0, 2.5, 3.0]
    })
    u2idx = {"u0": 0, "u1": 1, "u2": 2}
    i2idx = {"i0": 0, "i1": 1, "i2": 2}

    # build_matrix: nhị phân
    X_bin = build_matrix(df, u2idx, i2idx)
    assert X_bin.shape == (3, 3)
    assert np.all(X_bin.data == 1.0)
    assert X_bin[0, 0] == 1.0
    assert X_bin[0, 1] == 1.0

    # build_weighted: giá trị 0.0 phải bị loại bỏ (eliminate_zeros)
    X_w = build_weighted(df, df.weight, u2idx, i2idx)
    assert X_w.shape == (3, 3)
    assert X_w[0, 1] == 0.0  # trọng số 0 bị loại
    assert X_w[1, 1] == 2.5
    assert X_w[2, 2] == 3.0


def test_popularity_model():
    X = sp.csr_matrix([
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ], dtype=np.float32)
    pop = Popularity().fit(X)
    assert np.allclose(pop.pop, [3, 1, 1, 1])

    scores = pop.score([0, 1], X)
    assert scores.shape == (2, 4)
    assert np.allclose(scores[0], [3, 1, 1, 1])


def test_item_knn_properties():
    # Ma trận 4 users x 4 items
    X = sp.csr_matrix([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ], dtype=np.float32)

    knn = ItemKNN(k=2, shrink=5.0).fit(X)
    # Đường chéo S phải bằng 0 (item không gợi ý chính nó)
    assert np.all(knn.S.diagonal() == 0.0)

    # Mỗi hàng tối đa k láng giềng
    for i in range(knn.S.shape[0]):
        row_nnz = knn.S.indptr[i + 1] - knn.S.indptr[i]
        assert row_nnz <= 2

    # Shrink lớn hơn làm giảm tương đồng cosine
    knn_noshrink = ItemKNN(k=2, shrink=0.0).fit(X)
    knn_shrink = ItemKNN(k=2, shrink=10.0).fit(X)
    assert knn_noshrink.S.data.max() > knn_shrink.S.data.max()

    # Chấm điểm
    H = X[[0]]
    scores = knn.score([0], H)
    assert scores.shape == (1, 4)


def test_ease_properties():
    # EASE cần n_items nhỏ
    X = sp.csr_matrix([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 1, 1]
    ], dtype=np.float32)

    ease = EASE(lam=10.0).fit(X)
    # Đường chéo ma trận B bắt buộc phải bằng 0 (diagonal constraint)
    assert np.allclose(np.diag(ease.B), 0.0)
    assert ease.B.shape == (4, 4)

    # Điểm dự đoán
    H = X[[0]]
    scores = ease.score([0], H)
    assert scores.shape == (1, 4)


def test_pop_fallback():
    X = sp.csr_matrix([
        [1, 0, 0],
        [1, 1, 0],
    ], dtype=np.float32)
    # item 0 có độ phổ biến 2, item 1 có 1, item 2 có 0

    class DummyModel:
        def score(self, users, H):
            return np.zeros((len(users), 3), dtype=np.float32)

    fallback = PopFallback(DummyModel(), X, eps=1e-3)
    scores = fallback.score([0], X[[0]])
    # Item 0 phổ biến nhất -> điểm phá hòa cao nhất
    assert scores[0, 0] > scores[0, 1] > scores[0, 2]


def test_item_prior_rerank():
    class DummyModel:
        def score(self, users, H):
            return np.ones((len(users), 3), dtype=np.float32)

    prior_z = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    rerank = ItemPriorRerank(DummyModel(), prior_z, lam=0.5)

    scores = rerank.score([0], None)
    # z cao hơn thì điểm nhân exp(lam * z) phải cao hơn
    assert scores[0, 0] > scores[0, 1] > scores[0, 2]
    assert np.isclose(scores[0, 1], 1.0)  # exp(0) = 1
