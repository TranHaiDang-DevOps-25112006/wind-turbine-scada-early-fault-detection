"""
src/evaluation/evaluate_advanced.py
GHI CHÚ HỌC THUẬT: Số liệu minh họa giai đoạn phát triển, không dùng trong báo cáo cuối.
Báo cáo cuối cùng sử dụng 100% số liệu thực nghiệm từ module evaluate_real.py.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def main():
    print("[*] evaluate_advanced.py: Đang chạy tạo biểu đồ tham chiếu PR-Curve...")
    os.makedirs("reports/metrics", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    pred_path = "data/processed/prediction_scored.parquet"
    if not os.path.exists(pred_path):
        print(f"[!] Không tìm thấy {pred_path}, bỏ qua.")
        return

    df = pd.read_parquet(pred_path)
    t_start = pd.Timestamp("2023-08-05 06:10:00")
    t_end = pd.Timestamp("2023-08-19 06:10:00")
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    y = ((df['time_stamp'] >= t_start) & (df['time_stamp'] <= t_end)).astype(int).values

    thresholds = np.linspace(-0.06, 0.06, 25)
    records = []
    for thr in thresholds:
        pred = (df['anomaly_score'] >= thr).astype(int).values
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        records.append({
            'threshold': round(thr, 3), 'precision': round(p, 4),
            'recall': round(r, 4), 'f1_score': round(f1, 4),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        })

    sweep_df = pd.DataFrame(records)
    sweep_df.to_csv("reports/metrics/day11_threshold_sweep_metrics.csv", index=False)

    # Vẽ biểu đồ PR-Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    ax1.plot(sweep_df['recall'], sweep_df['precision'], 'o-', color='#1f77b4', lw=2)
    ax1.set_xlabel("Recall (Độ bao phủ)")
    ax1.set_ylabel("Precision (Độ chính xác)")
    ax1.set_title("Đường cong Precision - Recall (Tham chiếu)")
    ax1.grid(True, ls='--', alpha=0.5)

    ax2.plot(sweep_df['threshold'], sweep_df['precision'], label='Precision', color='green')
    ax2.plot(sweep_df['threshold'], sweep_df['recall'], label='Recall', color='orange')
    ax2.plot(sweep_df['threshold'], sweep_df['f1_score'], label='F1-Score', color='purple', ls='--')
    ax2.axvline(0.0, color='red', ls=':', label='Ngưỡng đề xuất (thr=0.0)')
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Chỉ số (0 - 1)")
    ax2.set_title("Đánh đổi Precision - Recall theo Threshold")
    ax2.legend()
    ax2.grid(True, ls='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig("reports/figures/day11_fig1_precision_recall_curve.png")
    plt.close(fig)
    print("[OK] Hoàn thành evaluate_advanced (Tham chiếu).")

if __name__ == "__main__":
    main()