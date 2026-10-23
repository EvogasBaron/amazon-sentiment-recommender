# Sentiment trong gợi ý sản phẩm (Amazon Reviews 2023, Video_Games)

Đồ án nghiên cứu ảnh hưởng của sentiment (từ bình luận) đối với thuật toán gợi ý.

## Cấu trúc và vai trò của từng tệp

| Đường dẫn / Tệp | Vai trò và chức năng |
|---|---|
| `src/config.py` | Cấu hình tập trung đường dẫn (`data/`, `results/`) và thiết lập danh mục phân tích (`RECSYS_CATEGORY`). |
| `src/split.py` | Chia dữ liệu Leave-one-out theo thời gian thực (train/valid/test) và cơ chế phá hòa tất định theo `parent_asin`. |
| `src/models.py` | Cài đặt các mô hình gợi ý: Popularity, ItemKNN (cosine + shrink), EASE (nghịch đảo ma trận tại chỗ), và Reranking. |
| `src/metrics.py` | Đánh giá xếp hạng toàn catalog (full ranking, không lấy mẫu âm) với cơ chế giải quyết hòa điểm bi quan (pessimistic ties). |
| `src/stats.py` | Kiểm định ý nghĩa thống kê bằng Paired Bootstrap Resampling, tính khoảng tin cậy 95% và hiệu chỉnh Bonferroni. |
| `src/sentiment_transformer.py` | Trích xuất điểm cảm xúc văn bản review theo lô bằng mô hình Pretrained Transformer (`nlptown/bert`), có lưu checkpoint. |
| `notebooks/prepare_and_eda.ipynb` | Tải dữ liệu Amazon Reviews 2023, lọc k-core, làm sạch tiêu đề tự sinh chống rò rỉ nhãn, chia tập dữ liệu và EDA. |
| `notebooks/sentiment_scoring.ipynb` | Trích xuất điểm cảm xúc bằng VADER (rule-based baseline) và tổng hợp bảng sentiment hợp nhất. |
| `notebooks/sentiment_transformer.ipynb` | Báo cáo chẩn đoán phân phối điểm sentiment từ mô hình Transformer và tương quan với số sao. |
| `notebooks/baselines.ipynb` | Huấn luyện, tinh chỉnh siêu tham số và đánh giá các mô hình nền tảng (Popularity, ItemKNN, EASE). |
| `notebooks/hybrid_ablation.ipynb` | Thực nghiệm 6 họ biến thể lai ghép sentiment (A-F), so sánh đối chứng với baseline bằng kiểm định thống kê. |
| `notebooks/analysis.ipynb` | Tổng hợp kết quả, trực quan hóa biểu đồ, phân tầng theo số sao/kỳ vọng và phân tích nguyên nhân gốc rễ. |
| `tests/test_core.py` | Unit test tự động kiểm tra tính chính xác của hàm split, xếp hạng ties, và paired bootstrap. |
| `requirements.txt` | Danh sách các thư viện phụ thuộc của môi trường thực thi. |

## Thứ tự chạy

1. `notebooks/prepare_and_eda.ipynb`: tải và lọc k-core, làm sạch tiêu đề tự sinh, chia train/valid/test, EDA, ghi `dataset_meta.json`.
2. `python -m src.sentiment_transformer score` rồi `python -m src.sentiment_transformer report` (chạy từ thư mục gốc dự án; cần `torch`, `transformers`). Xem `notebooks/sentiment_transformer.ipynb`.
3. `notebooks/sentiment_scoring.ipynb`: VADER (đường cơ sở) và tạo `sentiment_all.csv.gz`.
4. `notebooks/baselines.ipynb`: Popularity, ItemKNN, EASE; lưu `best_params.json`.
5. `notebooks/hybrid_ablation.ipynb`: các biến thể dùng sentiment, so với baseline bằng bootstrap ghép cặp.
6. `notebooks/analysis.ipynb`: tổng hợp, biểu đồ, phân tích sentiment so với số sao.

Mặc định danh mục là `Video_Games`. Đổi bằng biến môi trường `RECSYS_CATEGORY` (và `CATEGORY` trong `prepare_and_eda`), hai giá trị phải trùng nhau.

## Quy ước đánh giá

- Chia theo thời gian, leave-one-out: review cuối là test, kế cuối là valid, phần còn lại là train. Khi trùng timestamp, phá hòa theo `parent_asin` để kết quả ổn định.
- Xếp hạng toàn bộ item (không lấy mẫu âm), loại item user đã có trong lịch sử; ties xử lý bi quan.
- Cấu hình chọn trên valid, chấm test một lần. Mọi so sánh với baseline có khoảng tin cậy bootstrap ghép cặp (có thêm khoảng hiệu chỉnh Bonferroni).
- Văn bản chấm sentiment không chứa tiêu đề tự sinh của Amazon ("Five Stars"...), vì chuỗi này làm lộ số sao.
