# Thông tin Dữ liệu (Dataset Documentation)

Tài liệu mô tả nguồn gốc, kích thước, quy trình tiền xử lý và các đặc tính của tập dữ liệu sử dụng trong đồ án.

---

## 1. Nguồn dữ liệu & Kích thước

* **Tên tập dữ liệu:** Amazon Reviews 2023 (Danh mục `Video_Games`).
* **Nguồn tải chính thức:** [McAuley Lab - UCSD](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz)
* **Tệp dữ liệu gốc:** `Video_Games.jsonl.gz`
  * Dung lượng nén: ~814 MB
  * Không đưa trực tiếp vào repository Git vì vượt quá giới hạn lưu trữ của GitHub.
* **Quy trình lọc & Giảm tải tính toán:**
  * Lọc $k$-core ($k=9$): Loại bỏ người dùng và sản phẩm có ít hơn 9 tương tác để đảm bảo mật độ đồ thị kết nối và giảm thiểu chi phí tính toán khi inference mô hình Transformer.
  * Giới hạn: Sau khi lọc, tập dữ liệu đạt **257,475 reviews**, bao gồm **16,886 users** và **7,064 items**.

---

## 2. Tiền xử lý dữ liệu & Chống rò rỉ nhãn (Anti-Leakage)

1. **Làm sạch tiêu đề tự sinh của Amazon:**
   * Trong Amazon Reviews, khoảng 18.9% tiêu đề là các chuỗi do hệ thống tự sinh như *"Five Stars"*, *"One Star"*, *"Four Stars"*.
   * Toàn bộ các chuỗi này đã được loại bỏ trước khi đưa vào mô hình NLP để chống rò rỉ nhãn (target leakage), đảm bảo mô hình Transformer không "nhìn trộm" được số sao từ tiêu đề.
2. **Loại bỏ trùng lặp:**
   * Giữ lại review mới nhất cho mỗi cặp `(user_id, parent_asin)`.
3. **Phân chia dữ liệu theo dòng thời gian (Temporal Leave-One-Out):**
   * Review cuối cùng theo timestamp của mỗi user được dùng làm **Test Set** (16,886 reviews).
   * Review kế cuối được dùng làm **Validation Set** (16,886 reviews).
   * Toàn bộ các review trước đó làm **Training Set** (223,703 reviews).
   * Phá hòa tất định theo `parent_asin` khi trùng timestamp (tỷ lệ hòa điểm `tie_rate` cực thấp: 0.0001).

---

## 3. Giải thích về 28 review trống (Empty Text)

Trong quá trình trích xuất sentiment bằng mô hình Transformer (`nlptown/bert-base-multilingual-uncased-sentiment`) và VADER:
* Có **28 review** trong tổng số 257,475 review không có điểm cảm xúc (`tf01 = NaN`, `has_signal = False`, tỷ lệ chỉ **0.011%**).
* **Nguyên nhân:** Đây là các review mà người dùng chỉ để lại số sao mà hoàn toàn không viết tiêu đề lẫn nội dung bài viết.
* **Phân bố:** 28 review này tập trung ở 5 user. Phần lớn nằm ở tập Train.
* **Tác động đến Valid và Test:** Chỉ có đúng **2 review ở tập Valid** và **2 review ở tập Test** rơi vào trường hợp này (chiếm 0.012% mỗi tập).
* **Cơ chế xử lý:** Hệ thống gán giá trị trung tính $0.5$ ở bước hybrid, hoàn toàn không gây ảnh hưởng hay sai lệch đến kết luận của đồ án.

---

## 4. Tệp siêu dữ liệu đi kèm

* File [`data/processed/Video_Games/dataset_meta.json`](file:///d:/PycharmProjects/Sentiment_fix/data/processed/Video_Games/dataset_meta.json) được lưu trữ trực tiếp trên Git chứa đầy đủ các thông số chi tiết để kiểm chứng khả năng tái lập.
