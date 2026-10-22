import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from src.alerts.rules import apply_alert_rules

def load_event_info(file_path):
    df = pd.read_csv(file_path, sep=";")
    if "event_id" not in df.columns:
        df = pd.read_csv(file_path, sep=",")
    df.columns = [c.strip() for c in df.columns]
    return df

def run_evaluation(
    data_path="data/processed/prediction_scored.parquet",
    event_info_path="data/raw/Wind Farm A/event_info.csv",
    sample_event_path="data/sample/event_info.csv",
    output_metrics_dir="reports/metrics/",
    output_figures_dir="reports/figures/"
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[!] Không tìm thấy dữ liệu: {data_path}")
    df = pd.read_parquet(data_path)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df = df.sort_values("time_stamp").reset_index(drop=True)

    actual_event_path = sample_event_path if os.path.exists(sample_event_path) else event_info_path
    if not os.path.exists(actual_event_path):
        raise FileNotFoundError(f"[!] Không tìm thấy event_info.csv tại {actual_event_path}")
    
    event_df = load_event_info(actual_event_path)
    event_match = event_df[event_df["event_id"].astype(str) == "0"]
    event_0 = event_match.iloc[0] if not event_match.empty else event_df.iloc[0]

    t_start = pd.to_datetime(event_0["event_start"])
    t_end = pd.to_datetime(event_0["event_end"])
    desc = event_0.get("event_description", "Sự cố tua-bin")

    # Đồng bộ năm nếu lệch giữa nhãn và chuỗi thời gian
    data_year = df["time_stamp"].dt.year.mode()[0]
    if t_start.year != data_year:
        print(f"[!] Phát hiện lệch năm: Nhãn ({t_start.year}) vs Dữ liệu ({data_year}). Đang tự động đồng bộ...")
        year_diff = data_year - t_start.year
        t_start = t_start + pd.DateOffset(years=year_diff)
        t_end = t_end + pd.DateOffset(years=year_diff)

    print(f"[*] Đánh giá Sự kiện 0: {desc}")
    print(f" - Cửa sổ sự cố chuẩn hóa: {t_start} -> {t_end}")

    # Gán nhãn thực tế
    df["actual_label"] = ((df["time_stamp"] >= t_start) & (df["time_stamp"] <= t_end)).astype(int)

    # Áp dụng bộ lọc chống báo động giả
    df = apply_alert_rules(df, score_col="iso_label", consecutive=3, cooldown=6)
    df["raw_pred_label"] = (df["iso_label"] == -1).astype(int)

    prec = precision_score(df["actual_label"], df["raw_pred_label"], zero_division=0)
    rec = recall_score(df["actual_label"], df["raw_pred_label"], zero_division=0)
    f1 = f1_score(df["actual_label"], df["raw_pred_label"], zero_division=0)
    cm = confusion_matrix(df["actual_label"], df["raw_pred_label"])
    tn, fp, fn, tp = cm.ravel()

    # Tính Lead Time
    alerts_triggered = df[df["alert_flag"] == 1]
    lead_time_days = 0.0
    delay_from_start_days = 0.0
    first_alert_time = None

    if not alerts_triggered.empty:
        first_alert_time = alerts_triggered["time_stamp"].min()
        # Thời gian cảnh báo trước khi dừng máy (t_end)
        lead_time_days = (t_end - first_alert_time).total_seconds() / (24 * 3600)
        # Độ trễ nhận diện so với mốc chớm lỗi (t_start)
        delay_from_start_days = (first_alert_time - t_start).total_seconds() / (24 * 3600)

    print("\n" + "="*50)
    print(" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH BÁN GIÁM SÁT (CHUẨN ĐỒNG BỘ)")
    print("="*50)
    print(f" - Precision: {prec:.4f}")
    print(f" - Recall:    {rec:.4f}")
    print(f" - F1-Score:  {f1:.4f}")
    print(f" - Ma trận nhầm lẫn: TP={tp:,}, FP={fp:,}, TN={tn:,}, FN={fn:,}")
    print(f" - Còi cảnh báo kích hoạt: {df['alert_flag'].sum()}")
    if first_alert_time:
        print(f" - Thời điểm cảnh báo đầu tiên: {first_alert_time}")
        print(f" - Cảnh báo trước khi dừng máy (Lead Time): {lead_time_days:.2f} ngày")
        print(f" - Độ trễ phát hiện sau mốc chớm lỗi: {delay_from_start_days:.2f} ngày")
    print("="*50)

    # Xuất bảng metrics ra CSV
    os.makedirs(output_metrics_dir, exist_ok=True)
    metrics_path = os.path.join(output_metrics_dir, "day10_evaluation_metrics.csv")
    pd.DataFrame([{
        "event_id": 0,
        "event_description": desc,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "first_alert_time": str(first_alert_time),
        "lead_time_days": round(lead_time_days, 2),
        "delay_from_start_days": round(delay_from_start_days, 2)
    }]).to_csv(metrics_path, index=False)
    print(f"[✓] Đã xuất bảng chỉ số: {metrics_path}")

    # Vẽ và lưu biểu đồ Lead Time
    os.makedirs(output_figures_dir, exist_ok=True)
    fig_path = os.path.join(output_figures_dir, "day10_lead_time_event0.png")
    
    plt.figure(figsize=(14, 6), dpi=150)
    plt.plot(df["time_stamp"], df["anomaly_score"], label="Anomaly Score", color="#1f77b4", lw=1.2)
    plt.axvspan(t_start, t_end, color="red", alpha=0.25, label=f"Anomaly Window ({desc})")
    
    alert_points = df[df["alert_flag"] == 1]
    if not alert_points.empty:
        plt.scatter(alert_points["time_stamp"], alert_points["anomaly_score"], color="crimson", marker="^", s=70, zorder=5, label="Alert Triggered (3 chu kỳ)")

    plt.title(
        f"Phát hiện sớm sự cố Tua-bin gió (Event 0: {desc})\n"
        f"Lead Time báo trước khi dừng máy: {lead_time_days:.2f} ngày (Độ trễ sau chớm lỗi: {delay_from_start_days:.2f} ngày)",
        fontsize=12, fontweight="bold"
    )
    plt.xlabel("Thời gian (time_stamp)", fontsize=11)
    plt.ylabel("Anomaly Score (-decision_function)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"[✓] Đã xuất lại biểu đồ chuẩn: {fig_path}")

if __name__ == "__main__":
    run_evaluation()