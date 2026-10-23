import numpy as np


def sort_targets(target_df):
    """Thứ tự mà evaluate() dùng; dùng lại để gióng hàng rank với thuộc tính của item đích."""
    return target_df.sort_values("user_id").reset_index(drop=True)


def evaluate(model, H, target_df, user2idx, item2idx, ks=(5, 10, 20), batch=512):
    """
    Xếp hạng TOÀN BỘ item (không lấy mẫu), loại item user đã có trong H.
    Mỗi user có đúng 1 item đích. Ties xử lý bi quan (item bằng điểm xếp trước)
    để Popularity không được lợi giả. Trả về (dict metric, mảng rank từng user),
    rank sắp theo sort_targets(target_df).
    """
    t = sort_targets(target_df)
    users = t.user_id.map(user2idx).to_numpy()
    targets = t.parent_asin.map(item2idx).to_numpy()
    ranks = np.empty(len(t), dtype=np.int64)
    for s in range(0, len(t), batch):
        u, tg = users[s:s + batch], targets[s:s + batch]
        Hb = H[u]
        scores = model.score(u, Hb).astype(np.float32)
        r, c = Hb.nonzero()
        scores[r, c] = -np.inf
        ts = scores[np.arange(len(u)), tg]
        ranks[s:s + batch] = (scores >= ts[:, None]).sum(1) - 1
    out = {"n_eval": len(ranks)}
    for k in ks:
        hit = ranks < k
        out[f"HR@{k}"] = hit.mean()
        out[f"NDCG@{k}"] = np.where(hit, 1.0 / np.log2(ranks + 2), 0.0).mean()
    return out, ranks
