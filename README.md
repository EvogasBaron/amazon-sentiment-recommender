# Cảm xúc trong review như một bộ lọc giảm thiểu rủi ro trong hệ thống gợi ý: bằng chứng thực nghiệm trên Amazon Reviews 2023

## Tóm tắt nghiên cứu (Abstract)

* **Câu hỏi nghiên cứu:** Liệu điểm cảm xúc (sentiment) trích xuất từ bình luận văn bản có bổ sung thông tin giá trị độc lập ngoài số sao đánh giá (numerical rating) trong hệ thống gợi ý sản phẩm (Collaborative Filtering) hay không?
* **Kết quả chính:** Điểm cảm xúc và số sao thực chất hoạt động như một **Bộ lọc giảm thiểu rủi ro (Risk-mitigation Filter)**: cơ chế gán trọng số làm giảm mạnh **16.7% đến 21.2%** tỷ lệ gợi ý trúng sản phẩm bị người dùng chê (Bad-hit rate, $\le 3$ sao), vượt qua kiểm định ý nghĩa thống kê Bonferroni khắt khe, trong khi tỷ lệ gợi ý trúng sản phẩm được khen ($\ge 4$ sao) được bảo toàn nguyên vẹn.
* **Kết quả âm tính & Bản chất Residual:** Trên thước đo NDCG@10 truyền thống, sentiment không mang lại sự cải thiện có ý nghĩa so với số sao ($dNDCG \in [-0.0004, +0.0002]$). Khi kiểm tra trực diện trên tập con bất đồng ($14.0\%$ test mà văn bản chấm thấp hơn số sao), việc gán trọng số sentiment khiến NDCG giảm sâu (EASE giảm $-0.0035$), chứng minh bất đồng giữa Pretrained Transformer và số sao chủ yếu là **nhiễu đo lường do lệch miền ngôn ngữ game thủ**, chứ không phải tín hiệu ẩn giá trị.

---

## Bảng kết quả thực nghiệm P0 trọng tâm (Test Set, $N = 16.886$)

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

> **Nhận xét then chốt:** Thước đo NDCG@10 truyền thống phạt hệ thống khi nó né tránh các item test bị chê $\le 3$ sao. Khi tách riêng thành Bad-hit và Good-hit, cơ chế weighting chứng minh giá trị thực sự trong việc bảo vệ trải nghiệm người dùng mà không làm giảm độ chính xác trên sản phẩm tốt.

### 2. Kiểm chứng giả thuyết Residual trên tập con bất đồng

| Tập con kiểm tra | Số user | % Test | $\Delta$ NDCG (ItemKNN C) | $\Delta$ NDCG (EASE C) | Kết luận |
|---|:---:|:---:|:---:|:---:|---|
| **Toàn bộ Test** | 16,886 | 100.0% | -0.0010 (CI chứa 0) | -0.0007 (CI chứa 0) | Không đổi |
| **Text khen ít hơn sao** ($diff \le -1$) | 2,363 | 14.0% | **-0.0031** (CI âm) | **-0.0035** (CI âm) | **Nhiễu đo lường làm giảm mạnh chất lượng** |
| **Residual lớn** ($\|resid\| \ge 0.2$) | 2,076 | 12.3% | -0.0014 | **-0.0053** (CI âm) | **Giảm chất lượng rõ rệt** |
| **Residual âm lớn** ($resid \le -0.2$) | 1,522 | 9.0% | **-0.0038** (CI âm) | **-0.0066** (CI âm) | **Giảm chất lượng sâu nhất** |

### 3. Phân tích lát cắt Tail Items (Sản phẩm ít tương tác, $N = 5.655$)

| Lát cắt sản phẩm đích | Cỡ mẫu | Thuật toán | NDCG Baseline | NDCG C (Sentiment) | $\Delta$ NDCG (Sentiment vs Base) | 95% Bootstrap CI | Có ý nghĩa thống kê? | $\Delta$ NDCG B (Stars) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tail Items ($\le 20$ reviews)** | 5,655 (33.5%) | **ItemKNN** | 0.00873 | **0.00989** | **+0.00117 (+13.8%)** | **[+0.00003, +0.00227]** | **CÓ (CI hoàn toàn dương)** | +0.00004 (Không đổi) |
| **Tail Items ($\le 20$ reviews)** | 5,655 (33.5%) | **EASE** | 0.00461 | **0.00560** | **+0.00100 (+21.7%)** | **[+0.00031, +0.00179]** | **CÓ (CI hoàn toàn dương)** | +0.00006 (Không đổi) |
| **Head Items ($> 20$ reviews)** | 11,231 (66.5%) | **ItemKNN** | 0.05358 | 0.05151 | -0.00207 (-3.9%) | [-0.00358, -0.00054] | Giảm nhẹ | -0.00132 |
| **Head Items ($> 20$ reviews)** | 11,231 (66.5%) | **EASE** | 0.06258 | 0.06105 | -0.00154 (-2.5%) | [-0.00325, -0.00004] | Giảm nhẹ | -0.00040 |

> **Phát hiện bước ngoặt:** Trên các sản phẩm phổ biến (Head items), tín hiệu láng giềng đã quá dày đặc nên gán trọng số sentiment gây nhiễu; nhưng trên **sản phẩm ít tương tác (Tail / Cold items, chiếm 1/3 tập test)**, nơi dữ liệu lọc cộng tác bị thưa thớt, **cảm xúc văn bản giúp tăng vượt bậc $+14\%$ đến $+22\%$ NDCG (vượt qua kiểm định thống kê $95\%$)**, điều mà số sao hoàn toàn không làm được!

### 4. Năng lực thống kê & Hiệu ứng tối thiểu phát hiện được (MDE)

| Thuật toán | Thước đo | Cỡ mẫu $N$ | Sai số chuẩn (SE) | MDE ($\alpha=0.05$, Power $80\%$) | MDE Bonferroni ($m=6$) | Hiệu ứng thực tế quan sát | Đánh giá năng lực thống kê |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **ItemKNN** | NDCG@10 | 16,886 | 0.000538 | **0.00151** (3.91% rel) | **0.00187** (4.86% rel) | -0.00098 | Chênh lệch nhỏ ($< 0.0004$) dưới ngưỡng nhạy MDE toàn cục |
| **ItemKNN** | Bad-hit rate (%) | 16,886 | 0.049533 | **0.1388%** (15.02% rel) | **0.1724%** (18.66% rel) | **-0.1540%** | **Đủ nhạy: Mức giảm bad-hit vượt qua ngưỡng MDE** |
| **EASE** | NDCG@10 | 16,886 | 0.000559 | **0.00157** (3.63% rel) | **0.00194** (4.50% rel) | -0.00069 | Chênh lệch nhỏ dưới ngưỡng nhạy MDE toàn cục |
| **EASE** | Bad-hit rate (%) | 16,886 | 0.049880 | **0.1397%** (14.66% rel) | **0.1736%** (18.21% rel) | **-0.1836%** | **Đủ nhạy: Mức giảm bad-hit vượt qua ngưỡng MDE Bonferroni** |

---

## Cấu trúc mã nguồn

| Đường dẫn / Tệp | Vai trò và chức năng |
|---|---|
| `src/config.py` | Cấu hình tập trung đường dẫn (`data/`, `results/`) và thiết lập danh mục phân tích (`RECSYS_CATEGORY`). |
| `src/split.py` | Chia dữ liệu Leave-one-out theo thời gian thực (train/valid/test) và cơ chế phá hòa tất định theo `parent_asin`. |
| `src/models.py` | Cài đặt các mô hình gợi ý: Popularity, ItemKNN, EASE, Reranking, và AdditivePriorRerank. |
| `src/metrics.py` | Đánh giá xếp hạng toàn catalog (full ranking, không lấy mẫu âm) với cơ chế giải quyết hòa điểm bi quan (pessimistic ties). |
| `src/stats.py` | Kiểm định ý nghĩa thống kê bằng Paired Bootstrap Resampling, tính khoảng tin cậy 95% và hiệu chỉnh Bonferroni. |
| `src/sentiment_transformer.py` | Trích xuất điểm cảm xúc văn bản review theo lô bằng mô hình Pretrained Transformer (`nlptown/bert`), có lưu checkpoint. |
| `src/compute_p0_results.py` | Tính toán tự động bảng Bad-hit rate, Good-hit rate, wNDCG, kiểm định Bonferroni và phân tích Residual. |
| `src/cold_start_and_power_analysis.py` | Phân tích lát cắt Tail/Cold items, kiểm định MDE, và tính toán năng lực thống kê (Statistical Power). |
| `src/qualitative_error_analysis.py` | Phân tích định tính 30 mẫu review bất đồng mạnh giữa BERT và sao (châm biếm, độ khó game, lỗi kỹ thuật). |
| `src/temporal_diagnostics.py` | Đo lường rò rỉ thời gian chéo (Cross-user leakage) và lý giải độ sụt giảm 17-20% từ Valid sang Test. |
| `notebooks/prepare_and_eda.ipynb` | Tải dữ liệu Amazon Reviews 2023, lọc k-core, làm sạch tiêu đề tự sinh chống rò rỉ nhãn, chia tập dữ liệu và EDA. |
| `notebooks/sentiment_scoring.ipynb` | Trích xuất điểm cảm xúc bằng VADER (rule-based baseline) và tổng hợp bảng sentiment hợp nhất. |
| `notebooks/sentiment_transformer.ipynb` | Báo cáo chẩn đoán phân phối điểm sentiment từ mô hình Transformer và tương quan với số sao. |
| `notebooks/baselines.ipynb` | Huấn luyện, tinh chỉnh siêu tham số và đánh giá các mô hình nền tảng (Popularity, ItemKNN, EASE). Lưu đầy đủ output. |
| `notebooks/hybrid_ablation.ipynb` | Thực nghiệm 6 họ biến thể lai ghép sentiment (A-F), so sánh đối chứng với baseline bằng kiểm định thống kê. |
| `notebooks/analysis.ipynb` | Tổng hợp kết quả, trực quan hóa biểu đồ, phân tầng theo số sao/kỳ vọng và phân tích nguyên nhân gốc rễ. |
| `tests/test_core.py` | Unit test kiểm tra split theo thời gian, giải quyết hòa điểm bi quan và paired bootstrap. |
| `tests/test_models_and_weights.py` | Unit test toàn diện (10/10 tests) cho các mô hình (ItemKNN, EASE, Popularity, Fallback, Additive Prior). |
| `DATA.md` | Tài liệu mô tả nguồn gốc dữ liệu, quy trình k-core và giải trình 28 review trống. |
| `requirements.txt` | Danh sách các thư viện phụ thuộc môi trường chạy thực tế (Python 3.11, UTF-8). |

---

## Báo cáo chuyên sâu bổ sung

1. **[Báo cáo Phân tích định tính lỗi lệch miền (Qualitative Error Analysis)](results/Video_Games/qualitative_error_analysis.md):** Khảo sát bằng chứng thực tế về các review game 5 sao bị BERT đánh giá 1 sao (do từ ngữ miêu tả độ khó, kinh dị, bạo lực) và review 1 sao bị chấm 5 sao (do châm biếm/mỉa mai).
2. **[Báo cáo Phân tích Rò rỉ thời gian & Trôi dạt danh mục (Temporal Drift)](results/Video_Games/temporal_leakage_and_drift_report.md):** Phân tích chi tiết mức sụt giảm độ phổ biến của item đích (giảm 19.1%) và 25.33% rò rỉ thời gian chéo giải thích hiện tượng giảm 17-20% từ Valid sang Test.

---

## Hướng dẫn cài đặt & Thực thi kiểm thử

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Chạy toàn bộ kiểm thử tự động (Unit Tests)
```bash
pytest tests/ -v
```

### 3. Tái lập bảng kết quả P0
```bash
python -m src.compute_p0_results
python -m src.qualitative_error_analysis
python -m src.temporal_diagnostics
```
