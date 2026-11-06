"""
Tính toán các chỉ số P0 trọng tâm:
1. Bad-hit rate (% đích <= 3 sao lọt top 10)
2. Good-hit rate (% đích >= 4 sao lọt top 10)
3. wNDCG@10 (gain=0 cho đích <= 3 sao, gain=1 cho đích >= 4 sao)
4. Phân tích Residual và tập con bất đồng giữa văn bản (BERT) và số sao.
5. So sánh Valid vs Test (độ trôi thời gian, phân bố sao).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import RESULTS, SPLITS, SENT_FILE
from src.metrics import sort_targets
from src.stats import mean_ci

def run():
    print("=== ĐANG TÍNH TOÁN BẢNG CHỈ SỐ P0 ===")
    test_df = sort_targets(pd.read_csv(SPLITS / "test.csv.gz"))
    valid_df = sort_targets(pd.read_csv(SPLITS / "valid.csv.gz"))
    
    ratings_te = test_df.rating.to_numpy()
    is_bad = (ratings_te <= 3)
    is_good = (ratings_te >= 4)
    n_test = len(test_df)
    
    # 1. Bảng Bad-hit, Good-hit, wNDCG, NDCG trên Test
    rows = []
    for algo in ["knn", "ease"]:
        base_rk = np.load(RESULTS / f"ranks_hybrid_{algo}_A_baseline.npy")
        base_hit10 = (base_rk < 10)
        base_bad = (base_hit10 & is_bad).astype(float)
        base_good = (base_hit10 & is_good).astype(float)
        
        base_wndcg = np.where(base_hit10 & is_good, 1.0 / np.log2(base_rk + 2), 0.0)
        base_ndcg = np.where(base_hit10, 1.0 / np.log2(base_rk + 2), 0.0)
        
        rows.append({
            "algo": algo,
            "family": "A baseline",
            "bad_hit_%": 100 * base_bad.mean(),
            "good_hit_%": 100 * base_good.mean(),
            "wNDCG@10": base_wndcg.mean(),
            "NDCG@10": base_ndcg.mean(),
            "d_bad_hit_%": 0.0, "bad_lo": 0.0, "bad_hi": 0.0, "bad_sig_bonf": False,
            "d_good_hit_%": 0.0, "good_lo": 0.0, "good_hi": 0.0,
            "d_wNDCG": 0.0, "wndcg_lo": 0.0, "wndcg_hi": 0.0
        })
        
        algo_files = sorted(RESULTS.glob(f"ranks_hybrid_{algo}_*.npy"))
        for f in algo_files:
            fam_slug = f.stem.replace(f"ranks_hybrid_{algo}_", "")
            if fam_slug == "A_baseline":
                continue
            
            rk = np.load(f)
            hit10 = (rk < 10)
            bad = (hit10 & is_bad).astype(float)
            good = (hit10 & is_good).astype(float)
            wndcg = np.where(hit10 & is_good, 1.0 / np.log2(rk + 2), 0.0)
            ndcg = np.where(hit10, 1.0 / np.log2(rk + 2), 0.0)
            
            m_bad, lo_b, hi_b = mean_ci(100 * (bad - base_bad), B=2000, seed=0, alpha=0.05)
            _, lo_b_bf, hi_b_bf = mean_ci(100 * (bad - base_bad), B=2000, seed=0, alpha=0.05 / 6)
            
            m_good, lo_g, hi_g = mean_ci(100 * (good - base_good), B=2000, seed=0, alpha=0.05)
            m_w, lo_w, hi_w = mean_ci(wndcg - base_wndcg, B=2000, seed=0, alpha=0.05)
            
            sig_bonf = (lo_b_bf > 0 or hi_b_bf < 0)
            
            rows.append({
                "algo": algo,
                "family": fam_slug,
                "bad_hit_%": 100 * bad.mean(),
                "good_hit_%": 100 * good.mean(),
                "wNDCG@10": wndcg.mean(),
                "NDCG@10": ndcg.mean(),
                "d_bad_hit_%": m_bad, "bad_lo": lo_b, "bad_hi": hi_b, "bad_sig_bonf": sig_bonf,
                "d_good_hit_%": m_good, "good_lo": lo_g, "good_hi": hi_g,
                "d_wNDCG": m_w, "wndcg_lo": lo_w, "wndcg_hi": hi_w
            })
            
    df_p0 = pd.DataFrame(rows)
    df_p0.round(4).to_csv(RESULTS / "p0_metrics_test.csv", index=False)
    print("-> Đã lưu bảng p0_metrics_test.csv")

    # 2. Bảng Phân tích Residual & Tập con bất đồng
    print("\n=== ĐANG TÍNH TOÁN BẢNG RESIDUAL VÀ TẬP CON BẤT ĐỒNG ===")
    sent_all = pd.read_csv(SENT_FILE)
    test_sent = test_df.merge(sent_all[["user_id", "parent_asin", "tf01"]], on=["user_id", "parent_asin"], how="left")
    
    # Tính mean tf01 theo mức rating trên train+valid gộp
    train_df = pd.read_csv(SPLITS / "train.csv.gz")
    trva = pd.concat([train_df, valid_df])
    trva_sent = trva.merge(sent_all[["user_id", "parent_asin", "tf01"]], on=["user_id", "parent_asin"], how="left")
    mean_tf_by_rating = trva_sent.groupby("rating").tf01.mean().to_dict()
    
    test_sent["tf_exp_star"] = 1.0 + 4.0 * test_sent["tf01"]
    test_sent["diff"] = test_sent["tf_exp_star"] - test_sent["rating"]
    test_sent["resid"] = test_sent["tf01"] - test_sent["rating"].map(mean_tf_by_rating)
    
    subsets = {
        "Toàn bộ test": np.ones(len(test_df), dtype=bool),
        "Text khen ít hơn sao (diff <= -1.0)": (test_sent["diff"] <= -1.0).to_numpy(),
        "Text khen hơn sao (diff >= 1.0)": (test_sent["diff"] >= 1.0).to_numpy(),
        "Residual lớn (|resid| >= 0.2)": (test_sent["resid"].abs() >= 0.2).to_numpy(),
        "Residual âm lớn (resid <= -0.2)": (test_sent["resid"] <= -0.2).to_numpy(),
    }
    
    resid_rows = []
    for sub_name, mask in subsets.items():
        n_sub = int(mask.sum())
        pct_sub = 100.0 * n_sub / len(test_df)
        
        for algo in ["knn", "ease"]:
            rk_a = np.load(RESULTS / f"ranks_hybrid_{algo}_A_baseline.npy")[mask]
            rk_c = np.load(RESULTS / f"ranks_hybrid_{algo}_C_sentiment.npy")[mask]
            rk_d = np.load(RESULTS / f"ranks_hybrid_{algo}_D_stars_sentiment.npy")[mask]
            rk_b = np.load(RESULTS / f"ranks_hybrid_{algo}_B_stars.npy")[mask]
            
            ndcg_a = np.where(rk_a < 10, 1.0 / np.log2(rk_a + 2), 0.0)
            ndcg_c = np.where(rk_c < 10, 1.0 / np.log2(rk_c + 2), 0.0)
            ndcg_d = np.where(rk_d < 10, 1.0 / np.log2(rk_d + 2), 0.0)
            ndcg_b = np.where(rk_b < 10, 1.0 / np.log2(rk_b + 2), 0.0)
            
            dc, lo_c, hi_c = mean_ci(ndcg_c - ndcg_a, B=2000, seed=0)
            dd, lo_d, hi_d = mean_ci(ndcg_d - ndcg_a, B=2000, seed=0)
            db, lo_b, hi_b = mean_ci(ndcg_b - ndcg_a, B=2000, seed=0)
            
            resid_rows.append({
                "Tập con": sub_name,
                "n_user": n_sub,
                "% test": pct_sub,
                "Thuật toán": algo,
                "dNDCG C vs A": dc, "lo_C": lo_c, "hi_C": hi_c, "C_sig": (lo_c > 0 or hi_c < 0),
                "dNDCG D vs A": dd, "lo_D": lo_d, "hi_D": hi_d,
                "dNDCG B vs A": db, "lo_B": lo_b, "hi_B": hi_b,
            })
            
    df_resid = pd.DataFrame(resid_rows)
    df_resid.round(4).to_csv(RESULTS / "residual_analysis_test.csv", index=False)
    print("-> Đã lưu bảng residual_analysis_test.csv")
    print(df_resid[["Tập con", "n_user", "% test", "Thuật toán", "dNDCG C vs A", "lo_C", "hi_C", "C_sig"]].round(4).to_string())

if __name__ == "__main__":
    run()
