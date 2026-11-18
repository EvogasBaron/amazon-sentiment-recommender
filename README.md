# Cảm xúc trong review như một bộ lọc giảm thiểu rủi ro trong hệ thống gợi ý: bằng chứng thực nghiệm trên Amazon Reviews 2023

## Tóm tắt nghiên cứu (Abstract)

* **Câu hỏi nghiên cứu:** Liệu điểm cảm xúc (sentiment) trích xuất từ bình luận văn bản có bổ sung thông tin giá trị độc lập ngoài số sao đánh giá (numerical rating) trong hệ thống gợi ý sản phẩm (Collaborative Filtering) hay không?
* **Kết quả chính (Weighting theo phản hồi vs Nhị phân):** Đưa mức độ hài lòng (số sao hoặc điểm cảm xúc) vào mô hình gợi ý giúp tạo ra một **Bộ lọc giảm thiểu rủi ro (Risk-mitigation Filter)**: giảm mạnh **16.7% đến 21.2%** tỷ lệ gợi ý trúng sản phẩm bị người dùng chê (Bad-hit rate, $\le 3$ sao), vượt qua kiểm định ý nghĩa thống kê Bonferroni khắt khe, trong khi vẫn bảo toàn nguyên vẹn tỷ lệ gợi ý trúng sản phẩm được khen (Good-hit rate, $\ge 4$ sao) — một hiệu ứng tích cực bị NDCG@10 truyền thống phạt oan.
* **Kết quả âm tính (Sentiment vô hướng vs Số sao & Bản chất Residual):** Khi đối chứng trực tiếp giữa sentiment (`C`, `D`) và chỉ dùng số sao (`B`), **sentiment vô hướng từ Pretrained Transformer không mang lại lợi thế có ý nghĩa thống kê nào vượt trội số sao** trên bất kỳ thước đo nào (NDCG@10, wNDCG, Good-hit lẫn Bad-hit), với chênh lệch $dNDCG \in [-0.0004, +0.0002]$ (nhỏ hơn nhiều so với ngưỡng nhạy tối thiểu $\text{MDE} \approx 0.0015$). Khi kiểm tra trực diện trên tập con bất đồng ($14.0\%$ test mà văn bản chấm thấp hơn số sao), dùng sentiment làm NDCG giảm sâu (EASE giảm $-0.0035$), chứng minh bất đồng giữa BERT và số sao chủ yếu là **nhiễu đo lường do lệch miền ngôn ngữ game thủ**, chứ không phải tín hiệu ẩn giá trị.

---

## Bảng kết quả thực nghiệm trọng tâm (Test Set, $N = 16.886$)

### 1. Năng lực giảm thiểu rủi ro: Bad-hit vs Good-hit Rate

| Thuật toán | Biến thể | Bad-hit rate (%) | $\Delta$ Bad-hit (điểm %) | Ý nghĩa (Bonferroni) | Good-hit rate (%) | wNDCG@10 | NDCG@10 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ItemKNN** | A baseline (nhị phân) | 0.924% | - | - | 6.443% | 0.0342 | 0.0386 |
| | B stars | 0.841% | -0.083% | Không | 6.325% | 0.0339 | 0.0377 |
| | C sentiment (BERT) | **0.770%** | **-0.154%** | **Có** | 6.396% | 0.0339 | 0.0376 |
| | D stars + sentiment | 0.788% | -0.136% | Không | 6.497% | 0.0341 | 0.0378 |
| | E prior sentiment | **0.728%** | **-0.195%** | **Có** | 6.420% | 0.0337 | 0.0374 |
| **EASE** | A baseline (nhị phân) | 0.953% | - | - | 7.106% | 0.0385 | 0.0432 |
| | B stars | **0.794%** | **-0.160%** | **Có** | 7.148% | 0.0390 | 0.0429 |
| | C sentiment (BERT) | **0.770%** | **-0.184%** | **Có** | 7.118% | 0.0387 | 0.0425 |
| | D stars + sentiment | **0.788%** | **-0.166%** | **Có** | 7.172% | 0.0393 | 0.0431 |

> **Phân định 2 tầng kết luận:**
> 1. *So với Baseline nhị phân (`A`):* Cơ chế weighting theo số sao hoặc sentiment làm giảm rõ rệt Bad-hit rate (tránh gợi ý sản phẩm bị chê) mà không làm suy giảm Good-hit rate.
> 2. *Sentiment (`C`/`D`) so với Chỉ số sao (`B`):* Chênh lệch giữa `C`/`D` và `B` có khoảng tin cậy chứa 0 trên cả NDCG@10, wNDCG, Good-hit và Bad-hit (sau hiệu chỉnh Bonferroni). Điểm sentiment vô hướng không thêm thông tin mới ngoài số sao.

### 2. Kiểm chứng giả thuyết Residual trên tập con bất đồng

| Tập con kiểm tra | Số user | % Test | $\Delta$ NDCG (ItemKNN C) | $\Delta$ NDCG (EASE C) | Kết luận |
|---|:---:|:---:|:---:|:---:|---|
| **Toàn bộ Test** | 16,886 | 100.0% | -0.0010 (CI chứa 0) | -0.0007 (CI chứa 0) | Không đổi |
| **Text khen ít hơn sao** ($diff \le -1$) | 2,363 | 14.0% | **-0.0031** (CI âm) | **-0.0035** (CI âm) | **Nhiễu đo lường làm giảm mạnh chất lượng** |
| **Residual lớn** ($\|resid\| \ge 0.2$) | 2,076 | 12.3% | -0.0014 | **-0.0053** (CI âm) | **Giảm chất lượng rõ rệt** |
| **Residual âm lớn** ($resid \le -0.2$) | 1,522 | 9.0% | **-0.0038** (CI âm) | **-0.0066** (CI âm) | **Giảm chất lượng sâu nhất** |

### 3. Năng lực thống kê (MDE), Độ trôi thời gian & Lưu ý toán học

* **Năng lực thống kê (MDE):** Với $N = 16.886$, ngưỡng nhạy tối thiểu ở mức $\alpha=0.05$, công suất $80\%$ là $\text{MDE}_{95\%} \approx 0.00151$ trên NDCG@10 ($\text{MDE}_{\text{Bonf}} \approx 0.00187$). Chênh lệch giữa `C`/`D` và `B` ($< 0.0004$) nhỏ hơn nhiều so với MDE, trong khi mức giảm Bad-hit rate ($-0.154\%$ đến $-0.184\%$) vượt qua ngưỡng nhạy và đạt ý nghĩa thống kê sau Bonferroni.
* **Độ trôi thời gian Valid $\to$ Test:** Khoảng cách thời gian trung vị giữa Valid (`2017-06-13`) và Test (`2018-07-03`) là 13 tháng; độ phổ biến trung vị của item đích trong tập Train giảm $19.1\%$ (từ $47.0$ xuống $38.0$), giải thích phần lớn mức sụt giảm NDCG $16.8\% - 20.2\%$ từ Valid sang Test (phân bố sao chỉ giải thích $4-5\%$).
* **Lưu ý toán học về phương sai trọng số (`w / w.mean()`):** Khi chuẩn hóa trọng số theo trung bình cộng ($\mathbb{E}[w] = 1$) để giữ quy mô phần tử ngoài đường chéo của ma trận Gram $G = X_w^T X_w$, các phần tử trên đường chéo $G_{ii}$ vẫn bị nhân với $\mathbb{E}[w^2] = 1 + \text{Var}(w) > 1$. Đường chéo phình to theo $\text{Var}(w) \cdot \text{pop}(i)$ hoạt động như một bộ phạt độ phổ biến (anti-popularity regularization), đẩy điểm từ Head sang Tail items tỉ lệ với phương sai trọng số chứ không phải do ngữ nghĩa văn bản.

---

## Cấu trúc mã nguồn

| Đường dẫn / Tệp | Vai trò và chức năng |
|---|---|
| `src/config.py` | Cấu hình tập trung đường dẫn (`data/`, `results/`) và thiết lập danh mục phân tích (`RECSYS_CATEGORY`). |
| `src/split.py` | Chia dữ liệu Leave-one-out theo thời gian thực (train/valid/test) và cơ chế phá hòa tất định theo `parent_asin`. |
| `src/models.py` | Cài đặt các mô hình gợi ý: Popularity, ItemKNN (cosine + shrink), EASE (nghịch đảo ma trận tại chỗ), ItemPriorRerank và AdditivePriorRerank. |
| `src/metrics.py` | Đánh giá xếp hạng toàn catalog (full ranking, không lấy mẫu âm) với cơ chế giải quyết hòa điểm bi quan (pessimistic ties). |
| `src/stats.py` | Kiểm định ý nghĩa thống kê bằng Paired Bootstrap Resampling, tính khoảng tin cậy 95% và hiệu chỉnh Bonferroni. |
| `src/sentiment_transformer.py` | Trích xuất điểm cảm xúc văn bản review theo lô bằng mô hình Pretrained Transformer (`nlptown/bert`), có lưu checkpoint. |
| `notebooks/prepare_and_eda.ipynb` | Tải dữ liệu Amazon Reviews 2023, lọc k-core, làm sạch tiêu đề tự sinh chống rò rỉ nhãn, chia tập dữ liệu và EDA. |
| `notebooks/sentiment_scoring.ipynb` | Trích xuất điểm cảm xúc bằng VADER (rule-based baseline) và tổng hợp bảng sentiment hợp nhất. |
| `notebooks/sentiment_transformer.ipynb` | Báo cáo chẩn đoán phân phối điểm sentiment từ mô hình Transformer và tương quan với số sao. |
| `notebooks/baselines.ipynb` | Huấn luyện, tinh chỉnh siêu tham số và đánh giá các mô hình nền tảng (Popularity, ItemKNN, EASE), đã lưu đầy đủ output. |
| `notebooks/hybrid_ablation.ipynb` | Thực nghiệm 6 họ biến thể lai ghép sentiment (A-F), so sánh đối chứng với baseline bằng kiểm định thống kê. |
| `notebooks/analysis.ipynb` | Notebook tổng hợp toàn bộ kết quả: Baseline, Hybrid, Bad-hit/Good-hit/wNDCG, Residual tập bất đồng, MDE và Độ trôi thời gian. |
| `tests/test_core.py` | Unit test kiểm tra split theo thời gian, giải quyết hòa điểm bi quan và paired bootstrap. |
| `tests/test_models_and_weights.py` | Unit test cho các mô hình (ItemKNN, EASE, Popularity, PopFallback, ItemPriorRerank, AdditivePriorRerank) và ma trận trọng số. |
| `DATA.md` | Tài liệu mô tả nguồn gốc dữ liệu, quy trình k-core và giải trình 28 review trống. |
| `requirements.txt` | Danh sách các thư viện phụ thuộc môi trường chạy thực tế (Python 3.11, UTF-8). |

---

## Báo cáo chuyên sâu bổ sung

1. **[Báo cáo Phân tích định tính lỗi lệch miền (Qualitative Error Analysis)](results/Video_Games/qualitative_error_analysis.md):** Khảo sát bằng chứng thực tế về các review game 5 sao bị BERT đánh giá 1 sao (do từ ngữ miêu tả độ khó, kinh dị, bạo lực) và review 1 sao bị chấm 5 sao (do châm biếm/mỉa mai).
2. **[Báo cáo Phân tích Rò rỉ thời gian & Trôi dạt danh mục (Temporal Drift)](results/Video_Games/temporal_leakage_and_drift_report.md):** Phân tích chi tiết mức sụt giảm độ phổ biến của item đích (giảm 19.1%) và 25.33% rò rỉ thời gian chéo giải thích hiện tượng giảm 17-20% từ Valid sang Test.

---

## Thứ tự chạy & Kiểm thử

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Chạy toàn bộ kiểm thử tự động (Unit Tests)
```bash
pytest tests/ -v
```

### 3. Thứ tự chạy pipeline thực nghiệm
1. `notebooks/prepare_and_eda.ipynb`: tải và lọc k-core, làm sạch tiêu đề tự sinh, chia train/valid/test, EDA, ghi `dataset_meta.json`.
2. `python -m src.sentiment_transformer score` rồi `python -m src.sentiment_transformer report` (xem `notebooks/sentiment_transformer.ipynb`).
3. `notebooks/sentiment_scoring.ipynb`: VADER (đường cơ sở) và tạo `sentiment_all.csv.gz`.
4. `notebooks/baselines.ipynb`: Popularity, ItemKNN, EASE; lưu `best_params.json`.
5. `notebooks/hybrid_ablation.ipynb`: các biến thể dùng sentiment, so với baseline bằng bootstrap ghép cặp.
6. `notebooks/analysis.ipynb`: tổng hợp biểu đồ, Bad-hit/Good-hit/wNDCG, kiểm định Residual, MDE và phân tích độ trôi thời gian.
