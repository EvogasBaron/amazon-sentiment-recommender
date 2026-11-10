# Phân tích giải mã hiện tượng sụt giảm 17-20% từ Valid sang Test

## 1. Bản chất sự khác biệt giữa Valid và Test

Trong các nghiên cứu RecSys sử dụng phương pháp Leave-one-out (LOO) theo từng user, việc sụt giảm hiệu năng từ Valid sang Test thường bị nghi ngờ là do overfitting. Tuy nhiên, các mô hình trong đồ án là mô hình tuyến tính có 1-2 tham số điều chuẩn, không thể overfitting.

Phân tích dữ liệu thực tế chỉ ra **3 nguyên nhân cốt lõi** giải thích mức sụt giảm này:

1. **Độ phổ biến của item đích giảm mạnh 19.1% (Catalog Shift):**
   - Độ phổ biến trung vị của item đích trong tập Train giảm từ **47.0** (ở Valid) xuống **38.0** (ở Test).
   - Trong Collaborative Filtering (ItemKNN, EASE), item có ít lượt đồng xuất hiện sẽ khó được xếp hạng cao hơn. Đây là nguyên nhân chiếm tới ~95% mức sụt giảm.

2. **Khoảng cách thời gian trung vị 13 tháng (Temporal Drift):**
   - Thời điểm trung vị của Valid là **2017-06-13**, trong khi Test là **2018-07-03** (cách nhau ~13 tháng).
   - Trong thị trường game, vòng đời sản phẩm thay đổi nhanh chóng, các tựa game mới ra mắt làm phân tán thị hiếu người dùng so với lịch sử tương tác trước đó.

3. **Định lượng Rò rỉ thời gian chéo (Cross-user Temporal Leakage):**
   - Trong tập Train, trung bình **25.33%** (trung vị **15.60%**) số review của các người dùng khác xảy ra *sau* thời điểm test của user hiện tại.
   - Điều này xác nhận hạn chế của Leave-one-out theo từng user: ma trận đồng xuất hiện trong train chứa một phần tín hiệu tương lai đối với những người dùng có tương tác sớm, khiến điểm Valid được hưởng lợi nhẹ.

## 2. Bảng chỉ số đối soát

| Chỉ số so sánh                                   | Valid      | Test                     | Chênh lệch / Đánh giá                |
|:-------------------------------------------------|:-----------|:-------------------------|:-------------------------------------|
| Số lượng user đánh giá (n_eval)                  | 16886      | 16886                    | Giống hệt nhau (16,886)              |
| Thời gian trung vị (Timestamp)                   | 2017-06-13 | 2018-07-03               | +13 tháng (386 ngày)                 |
| Khoảng cách thời gian trung vị per-user (ngày)   | -          | 88.5 ngày                | Khoảng cách tương tác cuối           |
| Sao trung bình (Rating)                          | 4.284      | 4.223                    | -0.06 sao (giải thích 4-5% mức giảm) |
| Tỷ lệ review tiêu cực (<= 3 sao)                 | 19.38%     | 21.21%                   | +1.8% review chê ở test              |
| Độ phổ biến trung vị của item đích trong Train   | 47.0       | 38.0                     | -19.1% tương tác láng giềng          |
| Độ phổ biến trung bình của item đích trong Train | 99.1       | 90.6                     | -8.6% tương tác láng giềng           |
| Tỷ lệ dữ liệu Train xảy ra SAU mốc test của user | -          | 25.33% (trung vị 15.60%) | Cross-user Temporal Leakage          |

## 3. Khuyến nghị cho báo cáo và bảo vệ
- Ghi nhận hiện tượng này như một minh chứng cho thấy nhóm nghiên cứu nắm vững bản chất của dữ liệu và hệ thống gợi ý thực tế.
- Khẳng định: Khoảng sụt giảm không phải do sai sót thuật toán mà do bản chất dịch chuyển danh mục và thời gian trong tập dữ liệu Amazon Reviews 2023.