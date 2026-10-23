import pandas as pd

ORDER = ["user_id", "timestamp", "parent_asin"]   # parent_asin chỉ để phá hòa cho ổn định


def _sorted_with_rank(df: pd.DataFrame):
    """Sắp theo ORDER; rank 0 là review cuối của user, 1 là kế cuối."""
    d = df.sort_values(ORDER, kind="stable").reset_index(drop=True)
    return d, d.groupby("user_id").cumcount(ascending=False)


def leave_one_out(df: pd.DataFrame):
    """Review cuối của mỗi user -> test, kế cuối -> valid, còn lại -> train.

    Thứ tự sắp xếp có phá hòa theo parent_asin nên kết quả chia không phụ thuộc
    thứ tự dòng của file đầu vào.

    Giới hạn: mốc cắt là theo TỪNG user, không phải một thời điểm chung. Tương tác của
    user khác xảy ra sau thời điểm item đích vẫn nằm trong train, nên độ phổ biến, độ
    đồng xuất hiện và chất lượng item tính từ train có thể chứa thông tin "tương lai"
    so với item đích. Cần nêu giới hạn này khi báo cáo.
    """
    d, r = _sorted_with_rank(df)
    return d[r >= 2].copy(), d[r == 1].copy(), d[r == 0].copy()


def tie_rate(df: pd.DataFrame) -> float:
    """Tỉ lệ user có hai review cuối trùng timestamp (thứ tự test/valid khi đó là tùy ý)."""
    d, r = _sorted_with_rank(df)
    last = d[r == 0].set_index("user_id").timestamp
    prev = d[r == 1].set_index("user_id").timestamp
    return float((last.loc[prev.index] == prev).mean())
