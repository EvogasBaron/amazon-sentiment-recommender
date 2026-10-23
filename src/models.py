import numpy as np
import scipy.sparse as sp
from scipy.linalg import inv


def build_weighted(df, w, user2idx, item2idx):
    """Ma trận có trọng số; các tương tác trọng số 0 bị bỏ hẳn (dùng cho biến thể lọc tín hiệu âm)."""
    u = df.user_id.map(user2idx).to_numpy()
    i = df.parent_asin.map(item2idx).to_numpy()
    X = sp.csr_matrix((np.asarray(w, dtype=np.float32), (u, i)),
                      shape=(len(user2idx), len(item2idx)))
    X.eliminate_zeros()
    return X


def build_matrix(df, user2idx, item2idx):
    """Ma trận nhị phân user x item (implicit feedback)."""
    X = build_weighted(df, np.ones(len(df)), user2idx, item2idx)
    X.data[:] = 1.0
    return X


def _topk_per_row(S, k):
    """Giữ k phần tử lớn nhất của mỗi hàng."""
    S = S.tocsr()
    idx_parts, val_parts, ptr = [], [], [0]
    for i in range(S.shape[0]):
        s, e = S.indptr[i], S.indptr[i + 1]
        idx, v = S.indices[s:e], S.data[s:e]
        if len(v) > k:
            top = np.argpartition(-v, k)[:k]
            idx, v = idx[top], v[top]
        idx_parts.append(idx)
        val_parts.append(v)
        ptr.append(ptr[-1] + len(v))
    data = np.concatenate(val_parts) if val_parts else np.zeros(0, dtype=np.float32)
    indices = np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=np.int32)
    return sp.csr_matrix((data.astype(np.float32), indices, np.asarray(ptr)), shape=S.shape)


class Popularity:
    def fit(self, X):
        self.pop = np.asarray(X.sum(0)).ravel().astype(np.float32)
        return self

    def score(self, users, H):
        return np.tile(self.pop, (len(users), 1))


class ItemKNN:
    """Item-item cosine, giữ top-k láng giềng. shrink > 0 làm giảm độ tin cậy của cặp item ít đồng xuất hiện.

    weighted=True: X truyền vào fit là ma trận có trọng số, và khi chấm điểm lịch sử
    của user cũng lấy từ ma trận đó (H chỉ còn dùng để loại item đã có).
    """

    def __init__(self, k=50, shrink=0.0, weighted=False):
        self.k, self.shrink, self.weighted = k, shrink, weighted

    def fit(self, X):
        X = X.tocsr().astype(np.float32)
        self.Xw = X if self.weighted else None
        norms = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel()) + 1e-9
        C = (X.T @ X).tocoo()
        keep = C.row != C.col
        r, c, v = C.row[keep], C.col[keep], C.data[keep]
        v = v / (norms[r] * norms[c] + self.shrink)
        S = sp.csr_matrix((v.astype(np.float32), (r, c)), shape=C.shape)
        self.S = _topk_per_row(S, self.k)
        return self

    def score(self, users, H):
        return ((self.Xw[users] if self.weighted else H) @ self.S).toarray()


class EASE:
    """EASE (closed-form linear autoencoder): baseline mạnh, không cần thư viện ngoài.
    Cần ma trận item x item dày nên chỉ dùng khi số item không quá lớn.
    weighted=True: như ItemKNN, lịch sử khi chấm điểm lấy từ ma trận trọng số đã fit."""

    def __init__(self, lam=500.0, max_items=20000, weighted=False):
        self.lam, self.max_items, self.weighted = lam, max_items, weighted

    def fit(self, X):
        X = X.tocsr().astype(np.float32)
        self.Xw = X if self.weighted else None
        n = X.shape[1]
        if n > self.max_items:
            raise ValueError(f"EASE cần ma trận {n}x{n}; vượt max_items={self.max_items}")
        dtype = np.float64 if n <= 8000 else np.float32
        G = (X.T @ X).toarray().astype(dtype)
        G[np.diag_indices(n)] += self.lam
        P = inv(G, overwrite_a=True, check_finite=False)   # nghịch đảo tại chỗ: đỉnh bộ nhớ ~1,5 n^2 thay vì ~3,5 n^2
        del G
        P /= -np.diag(P).copy()[None, :]
        np.fill_diagonal(P, 0.0)
        self.B = P.astype(np.float32)
        return self

    def score(self, users, H):
        return np.asarray((self.Xw[users] if self.weighted else H) @ self.B)


class PopFallback:
    """Cộng một lượng rất nhỏ theo độ phổ biến để phá hòa giữa các item cùng điểm 0
    (user có lịch sử ngắn thường có rất nhiều item điểm 0)."""

    def __init__(self, base, X, eps=1e-4):
        pop = np.asarray(X.sum(0)).ravel().astype(np.float32)
        self.base = base
        self.pop = (eps * pop / max(float(pop.max()), 1.0)).astype(np.float32)

    def score(self, users, H):
        return self.base.score(users, H) + self.pop[None, :]


class ItemPriorRerank:
    """Nhân điểm của model gốc với exp(lam * z_item), z_item là điểm chất lượng item đã chuẩn hóa."""

    def __init__(self, base, prior_z, lam):
        self.base, self.w = base, np.exp(lam * prior_z).astype(np.float32)

    def score(self, users, H):
        return self.base.score(users, H) * self.w[None, :]
