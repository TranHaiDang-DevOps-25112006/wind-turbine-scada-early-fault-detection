"""
Mục đích: Đánh giá thực nghiệm Event 0 (Wind Farm A) hoàn toàn từ dữ liệu thật, thay cho các
          số liệu viết sẵn trong evaluate_advanced.py / evaluate_multievent.py.

Script này KHÔNG chứa số liệu kết quả viết tay. Mọi con số đều được tính từ:
  - data/processed/train_scaled.parquet, prediction_scaled.parquet  (đầu vào mô hình)
  - data/processed/train_clean_normal.parquet, prediction_clean.parquet (cho baseline Z-Score)
  - data/processed/prediction_scored.parquet (kết quả train.py, seed 42, để đối chiếu)
  - data/sample/sample_event_0.csv, data/sample/event_info.csv (để xác định cửa sổ sự cố theo id)

Cách chạy (từ thư mục gốc dự án):
    python -m src.evaluation.evaluate_real
    python -m src.evaluation.evaluate_real --n-seeds 30

Đầu ra:
    reports/metrics/real_seed_stability.csv        Kết quả từng seed
    reports/metrics/real_seed_summary.csv          Thống kê tổng hợp theo seed
    reports/metrics/real_baseline_zscore.csv       Baseline Z-Score trên một số cảm biến
    reports/metrics/real_daily_anomaly_rate.csv    Tỉ lệ điểm bất thường theo ngày
    reports/metrics/real_run_info.csv              Thông tin chạy (số dòng, mốc sự kiện, kiểm tra tái lập)
    reports/figures/real_fig1_score_timeline.png
    reports/figures/real_fig2_daily_anomaly_rate.png
    reports/figures/real_fig3_seed_stability.png

Quy ước thuật ngữ: kết quả là "cảnh báo bất thường / cần kiểm tra", không phải "hỏng hóc chắc chắn".
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.alerts.rules import apply_alert_rules  # noqa: E402

PROCESSED = ROOT_DIR / "data" / "processed"
SAMPLE_DIR = ROOT_DIR / "data" / "sample"
METRICS = ROOT_DIR / "reports" / "metrics"
FIGURES = ROOT_DIR / "reports" / "figures"

META_COLS = ["time_stamp", "asset_id", "status_type_id"]
EVENT_ID = 0
BASELINE_SENSORS = ["sensor_6_avg", "sensor_13_avg", "sensor_14_avg"]
MODEL_PARAMS = dict(n_estimators=200, contamination=0.01, max_samples="auto")
THRESHOLD = 0.0
CONSECUTIVE = 3
COOLDOWN = 6


def load_event_window():
    """Xác định cửa sổ sự cố theo id bản ghi (event_start_id, event_end_id)."""
    info = pd.read_csv(SAMPLE_DIR / "event_info.csv", sep=";")
    if "event_id" not in info.columns:
        info = pd.read_csv(SAMPLE_DIR / "event_info.csv", sep=",")
    info.columns = [c.strip() for c in info.columns]
    row = info[info["event_id"].astype(str) == str(EVENT_ID)].iloc[0]

    raw = pd.read_csv(SAMPLE_DIR / "sample_event_0.csv", sep=";")
    raw["time_stamp"] = pd.to_datetime(raw["time_stamp"])
    id2ts = raw.set_index("id")["time_stamp"]
    t_start = id2ts.loc[int(row["event_start_id"])]
    t_end = id2ts.loc[int(row["event_end_id"])]
    return {
        "event_id": EVENT_ID,
        "description": row.get("event_description", ""),
        "label_start_in_event_info": str(row["event_start"]),
        "label_end_in_event_info": str(row["event_end"]),
        "t_start": t_start,
        "t_end": t_end,
    }


def load_data():
    tr = pd.read_parquet(PROCESSED / "train_scaled.parquet")
    pr = pd.read_parquet(PROCESSED / "prediction_scaled.parquet")
    tr_clean = pd.read_parquet(PROCESSED / "train_clean_normal.parquet")
    pr_clean = pd.read_parquet(PROCESSED / "prediction_clean.parquet")
    saved = pd.read_parquet(PROCESSED / "prediction_scored.parquet")
    for d in (tr, pr, tr_clean, pr_clean, saved):
        d["time_stamp"] = pd.to_datetime(d["time_stamp"])
    return tr, pr, tr_clean, pr_clean, saved


def confusion_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def days(delta):
    return round(delta.total_seconds() / 86400.0, 2)


def alert_summary(ts, alerts, t_start, t_end):
    """Tóm tắt cảnh báo theo cấp sự kiện."""
    a_ts = ts[alerts == 1]
    before = int((a_ts < t_start).sum())
    inside = int(((a_ts >= t_start) & (a_ts <= t_end)).sum())
    after = int((a_ts > t_end).sum())
    first_any = a_ts.min() if len(a_ts) else pd.NaT
    in_win = a_ts[(a_ts >= t_start) & (a_ts <= t_end)]
    first_in = in_win.min() if len(in_win) else pd.NaT
    return {
        "n_alerts": int(len(a_ts)),
        "alerts_before_start": before,
        "alerts_in_window": inside,
        "alerts_after_end": after,
        "first_alert_any": first_any,
        "delay_from_label_start_days": days(first_any - t_start) if pd.notna(first_any) else np.nan,
        "first_alert_in_window": first_in,
        "delay_in_window_days": days(first_in - t_start) if pd.notna(first_in) else np.nan,
        "time_to_label_end_days": days(t_end - first_in) if pd.notna(first_in) else np.nan,
    }


def run_one_seed(seed, tr, pr, feat_cols, y, ts, t_start, t_end, status0_mask):
    model = IsolationForest(random_state=seed, n_jobs=-1, **MODEL_PARAMS)
    model.fit(tr[feat_cols])
    score = -model.decision_function(pr[feat_cols])
    raw = (score >= THRESHOLD).astype(int)
    alerts = apply_alert_rules(pd.Series(score), THRESHOLD, CONSECUTIVE, COOLDOWN)["alert_triggered"].values

    row = {"seed": seed}
    m_all = confusion_metrics(y, raw)
    m_s0 = confusion_metrics(y[status0_mask], raw[status0_mask])
    row.update({f"raw_{k}": v for k, v in m_all.items()})
    row.update({f"raw_status0_{k}": v for k, v in m_s0.items()})
    row.update(alert_summary(ts, alerts, t_start, t_end))
    return row, score, alerts


def baseline_zscore(tr_clean, pr_clean, y, ts, t_start, t_end):
    """Baseline thống kê: |x - mu| / sigma > 3 với mu, sigma từ train bình thường."""
    rows = []
    for col in BASELINE_SENSORS:
        if col not in tr_clean.columns:
            continue
        mu, sd = float(tr_clean[col].mean()), float(tr_clean[col].std())
        if not sd or np.isnan(sd):
            continue
        z = (pr_clean[col] - mu).abs() / sd
        pred = (z > 3.0).astype(int).values
        m = confusion_metrics(y, pred)
        al = apply_alert_rules(pd.Series(z.values), threshold=3.0,
                               consecutive_points=CONSECUTIVE, cooldown_points=COOLDOWN)["alert_triggered"].values
        rows.append({"sensor": col, "train_mean": round(mu, 3), "train_std": round(sd, 3), "z_threshold": 3.0,
                     **{f"raw_{k}": v for k, v in m.items()},
                     **alert_summary(ts, al, t_start, t_end)})
    return pd.DataFrame(rows)


def fig_timeline(ts, score, alerts, t_start, t_end, path):
    fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
    ax.plot(ts, score, lw=1.0, color="#1f77b4", label="Anomaly score (-decision_function), seed 42")
    ax.axhline(THRESHOLD, color="gray", ls="--", lw=1, label=f"Ngưỡng = {THRESHOLD}")
    ax.axvspan(t_start, t_end, color="#ff9999", alpha=0.3, label="Cửa sổ sự cố theo nhãn (Event 0)")
    a = alerts == 1
    ax.scatter(ts[a], score[a], marker="^", color="crimson", s=40, zorder=5,
               label=f"Cảnh báo (≥{CONSECUTIVE} kỳ liên tiếp, cooldown {COOLDOWN})")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Anomaly score")
    ax.set_title("Anomaly score trên tập prediction – Event 0 (tính từ dữ liệu thật)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_daily_rate(daily, t_start, t_end, path):
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
    x = pd.to_datetime(daily["date"])
    ax.bar(x, daily["raw_anomaly_rate"] * 100, width=0.8, color="#ff7f0e")
    ax.axvspan(t_start, t_end, color="#ff9999", alpha=0.25, label="Cửa sổ sự cố theo nhãn")
    ax.set_ylabel("Tỉ lệ bản ghi vượt ngưỡng (%)")
    ax.set_xlabel("Ngày")
    ax.set_title("Tỉ lệ điểm bất thường theo ngày (seed 42, ngưỡng 0.0)")
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_seed_stability(df, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    s = df.sort_values("seed")
    x = np.arange(len(s))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in s["delay_from_label_start_days"]]
    ax1.bar(x, s["delay_from_label_start_days"], color=colors)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(s["seed"], fontsize=7, rotation=90)
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Cảnh báo đầu tiên − mốc bắt đầu nhãn (ngày)")
    ax1.set_title("Thời điểm cảnh báo đầu tiên theo seed\n(đỏ: trước mốc nhãn, xanh: sau mốc nhãn)")
    ax1.grid(True, axis="y", ls="--", alpha=0.4)
    ax2.plot(x, s["raw_precision"], "o-", label="Precision")
    ax2.plot(x, s["raw_recall"], "s-", label="Recall")
    ax2.plot(x, s["raw_f1"], "^-", label="F1")
    ax2.set_xticks(x)
    ax2.set_xticklabels(s["seed"], fontsize=7, rotation=90)
    ax2.set_xlabel("Seed")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Chỉ số theo bản ghi (dự đoán thô, ngưỡng 0.0) theo seed")
    ax2.legend()
    ax2.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20, help="Số seed (0..n-1), luôn gồm cả seed 42")
    args = parser.parse_args()

    METRICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    ev = load_event_window()
    tr, pr, tr_clean, pr_clean, saved = load_data()
    t_start, t_end = ev["t_start"], ev["t_end"]

    feat_cols = [c for c in tr.columns if c not in META_COLS]
    ts = pr["time_stamp"].reset_index(drop=True)
    y = ((ts >= t_start) & (ts <= t_end)).astype(int).values
    status0 = (pr["status_type_id"].values == 0)

    print("=" * 70)
    print(f"Event {ev['event_id']}: {ev['description']}")
    print(f"  Nhãn trong event_info.csv : {ev['label_start_in_event_info']} -> {ev['label_end_in_event_info']}")
    print(f"  Cửa sổ theo id (dùng ở đây): {t_start} -> {t_end}")
    print(f"  Train (status=0): {len(tr)} dòng | Prediction: {len(pr)} dòng | Đặc trưng: {len(feat_cols)}")
    print(f"  Bản ghi trong cửa sổ sự cố: {int(y.sum())} ({y.mean()*100:.1f}% tập prediction)")

    seeds = sorted(set(range(args.n_seeds)) | {42})
    results, score42, alerts42 = [], None, None
    for sd in seeds:
        row, score, alerts = run_one_seed(sd, tr, pr, feat_cols, y, ts, t_start, t_end, status0)
        results.append(row)
        if sd == 42:
            score42, alerts42 = score, alerts
    saved_sorted = saved.sort_values("time_stamp").reset_index(drop=True)
    max_diff = float(np.max(np.abs(score42 - saved_sorted["anomaly_score"].values)))
    print(f"\n[Kiểm tra tái lập] Sai khác tối đa giữa seed 42 và prediction_scored.parquet: {max_diff:.2e}")

    df = pd.DataFrame(results)
    df.to_csv(METRICS / "real_seed_stability.csv", index=False)

    cols = ["raw_precision", "raw_recall", "raw_f1", "raw_fp", "n_alerts",
            "alerts_before_start", "delay_from_label_start_days", "delay_in_window_days",
            "time_to_label_end_days"]
    summ = df[cols].agg(["mean", "std", "min", "max"]).T.round(3)
    summ.index.name = "metric"
    summ.to_csv(METRICS / "real_seed_summary.csv")
    n_early = int((df["alerts_before_start"] > 0).sum())
    print(f"\n[Độ ổn định theo {len(df)} seed]")
    print(summ.to_string())
    print(f"Số seed có cảnh báo xuất hiện TRƯỚC mốc bắt đầu nhãn: {n_early}/{len(df)}")

    base = baseline_zscore(tr_clean, pr_clean.sort_values("time_stamp").reset_index(drop=True),
                           y, ts, t_start, t_end)
    base.to_csv(METRICS / "real_baseline_zscore.csv", index=False)
    print("\n[Baseline Z-Score (ngưỡng 3σ, thống kê từ train bình thường)]")
    show = ["sensor", "raw_tp", "raw_fp", "raw_precision", "raw_recall", "raw_f1", "n_alerts"]
    print(base[show].to_string(index=False))

    raw42 = (score42 >= THRESHOLD).astype(int)
    daily = (pd.DataFrame({"date": ts.dt.date, "raw": raw42, "status0": status0.astype(int)})
             .groupby("date").agg(n_records=("raw", "size"),
                                  raw_anomaly_rate=("raw", "mean"),
                                  status0_share=("status0", "mean")).round(3).reset_index())
    daily.to_csv(METRICS / "real_daily_anomaly_rate.csv", index=False)

    r42 = df[df["seed"] == 42].iloc[0]
    info = pd.DataFrame([{
        "event_id": ev["event_id"], "description": ev["description"],
        "t_start": t_start, "t_end": t_end,
        "n_train": len(tr), "n_prediction": len(pr), "n_features": len(feat_cols),
        "n_positive_records": int(y.sum()), "positive_share": round(float(y.mean()), 4),
        "n_stopped_state_records_in_prediction": int((~status0).sum()),
        "seed42_vs_saved_max_abs_diff": max_diff,
        "n_seeds": len(df), "n_seeds_with_alert_before_start": n_early,
    }])
    info.to_csv(METRICS / "real_run_info.csv", index=False)

    fig_timeline(ts, score42, alerts42, t_start, t_end, FIGURES / "real_fig1_score_timeline.png")
    fig_daily_rate(daily, t_start, t_end, FIGURES / "real_fig2_daily_anomaly_rate.png")
    fig_seed_stability(df, FIGURES / "real_fig3_seed_stability.png")

    print("\n[Seed 42 – đối chiếu với số liệu Day 10]")
    print(f"  Precision={r42['raw_precision']}, Recall={r42['raw_recall']}, F1={r42['raw_f1']} "
          f"(TP={r42['raw_tp']}, FP={r42['raw_fp']}, TN={r42['raw_tn']}, FN={r42['raw_fn']})")
    print("\nLưu ý khi diễn giải:")
    print("  - Đánh giá theo bản ghi dùng nhãn theo cửa sổ thời gian (proxy), chỉ trên 1 sự kiện.")
    print("  - Tập train chỉ có vài chục giờ dữ liệu; kết quả phụ thuộc seed (xem real_seed_summary.csv).")
    print("  - 'time_to_label_end_days' không phải thời gian báo trước hỏng hóc.")
    print("[OK] Đã ghi kết quả vào reports/metrics/ và reports/figures/")


if __name__ == "__main__":
    main()