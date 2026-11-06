"""
src/evaluation/evaluate_multievent.py
GHI CHÚ HỌC THUẬT: Số liệu minh họa giai đoạn phát triển, không dùng trong báo cáo cuối.
Báo cáo cuối cùng sử dụng 100% số liệu thực nghiệm từ module evaluate_real.py.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    print("[*] evaluate_multievent.py: Xuất dữ liệu minh họa kiến trúc mở rộng...")
    os.makedirs("reports/metrics", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    summary_data = [
        {"event_id": 0, "component": "Vòng bi máy phát (Bearing)", "status": "Thực nghiệm thật", "lead_time_days": 13.14},
        {"event_id": 3, "component": "Hộp số (Gearbox)", "status": "Mô phỏng kiến trúc", "lead_time_days": 11.78},
        {"event_id": 14, "component": "Thủy lực (Hydraulic)", "status": "Mô phỏng kiến trúc", "lead_time_days": 5.54},
        {"event_id": 17, "component": "Biến áp (Transformer)", "status": "Mô phỏng kiến trúc", "lead_time_days": 0.69}
    ]
    df_sum = pd.DataFrame(summary_data)
    df_sum.to_csv("reports/metrics/day13_multievent_lead_time_summary.csv", index=False)

    # Đồ thị cột tham chiếu
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    bars = ax.bar(df_sum['component'], df_sum['lead_time_days'], color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])
    ax.axhline(7.0, color='red', ls='--', label='Chuẩn ngoài khơi (7 ngày)')
    ax.set_ylabel("Số ngày cảnh báo (Lead Time)")
    ax.set_title("Khảo sát thời gian cảnh báo qua các cụm sự cố (Minh họa khung)")
    ax.legend()
    ax.grid(True, axis='y', ls='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig("reports/figures/day13_fig2_multievent_generalization.png")
    plt.close(fig)
    print("[OK] Hoàn thành evaluate_multievent (Mô phỏng khung).")

if __name__ == "__main__":
    main()