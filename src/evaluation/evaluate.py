"""
src/evaluation/evaluate.py
Đánh giá độc lập mô hình Isolation Forest trên Event 0:
Tách bạch tầng điểm số thô (Frame-level) và tầng cảnh báo đã lọc (Alert-level).
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src.alerts.rules import apply_alert_rules

def evaluate_event0():
    pred_path = "data/processed/prediction_scored.parquet"
    if not os.path.exists(pred_path):
        print(f"[ERROR] Không tìm thấy {pred_path}. Hãy chạy tiền xử lý và train trước!")
        return

    df = pd.read_parquet(pred_path)
    
    event_start = pd.Timestamp("2023-08-05 06:10:00")
    event_end = pd.Timestamp("2023-08-19 06:10:00")
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df['ground_truth'] = ((df['time_stamp'] >= event_start) & (df['time_stamp'] <= event_end)).astype(int)

    # Đánh giá tầng dự đoán thô (Frame-level)
    df['raw_pred'] = (df['iso_label'] == -1).astype(int)
    p_raw = precision_score(df['ground_truth'], df['raw_pred'], zero_division=0)
    r_raw = recall_score(df['ground_truth'], df['raw_pred'], zero_division=0)
    f1_raw = f1_score(df['ground_truth'], df['raw_pred'], zero_division=0)
    tn_r, fp_r, fn_r, tp_r = confusion_matrix(df['ground_truth'], df['raw_pred']).ravel()

    # Đánh giá tầng cảnh báo sau lọc (Alert-level)
    alert_res = apply_alert_rules(df['anomaly_score'], threshold=0.0, consecutive_points=3, cooldown_points=6)
    df['alert_triggered'] = alert_res['alert_triggered']
    
    alert_points = df[df['alert_triggered'] == 1]
    if len(alert_points) > 0:
        first_alert = alert_points['time_stamp'].min()
        detection_delay_days = (first_alert - event_start).total_seconds() / 86400.0
        time_to_stoppage_days = (event_end - first_alert).total_seconds() / 86400.0
    else:
        first_alert = None
        detection_delay_days = 0.0
        time_to_stoppage_days = 0.0

    os.makedirs("reports/metrics", exist_ok=True)
    metrics_data = [{
        "event_id": 0,
        "precision_raw": round(p_raw, 4),
        "recall_raw": round(r_raw, 4),
        "f1_raw": round(f1_raw, 4),
        "tp_raw": int(tp_r),
        "fp_raw": int(fp_r),
        "tn_raw": int(tn_r),
        "fn_raw": int(fn_r),
        "total_filtered_alerts": int(df['alert_triggered'].sum()),
        "first_alert_time": str(first_alert),
        "detection_delay_days": round(detection_delay_days, 2),
        "lead_time_to_stoppage_days": round(time_to_stoppage_days, 2)
    }]
    
    res_df = pd.DataFrame(metrics_data)
    res_df.to_csv("reports/metrics/day10_evaluation_metrics.csv", index=False, encoding='utf-8-sig')
    print("[OK] Đã hoàn thành đánh giá evaluate.py và cập nhật reports/metrics/day10_evaluation_metrics.csv")

if __name__ == "__main__":
    evaluate_event0()