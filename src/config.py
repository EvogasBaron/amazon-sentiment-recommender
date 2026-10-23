"""Cấu hình đường dẫn dùng chung cho mọi notebook và script.

Đổi danh mục bằng biến môi trường RECSYS_CATEGORY (mặc định Video_Games),
phải trùng với CATEGORY trong notebook prepare_and_eda.
"""
import os
from pathlib import Path

CATEGORY = os.environ.get("RECSYS_CATEGORY", "Video_Games")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / CATEGORY
CORE_FILE = DATA / f"{CATEGORY}_core.csv.gz"
SPLITS = DATA / "splits"
SENT_FILE = DATA / "sentiment_all.csv.gz"
RESULTS = ROOT / "results" / CATEGORY
RESULTS.mkdir(parents=True, exist_ok=True)
