import numpy as np
import pandas as pd

def apply_alert_rules(df, score_col="iso_label", consecutive=3, cooldown=6):
    """
    Áp dụng quy tắc lọc chống báo động giả:
    - consecutive: Cần ít nhất N chu kỳ liên tiếp bất thường (iso_label == -1)
    - cooldown: Tạm khóa cảnh báo lặp lại trong M chu kỳ sau khi đã kích hoạt
    """
    df = df.copy()
    is_anomaly = (df[score_col] == -1).astype(int).values
    n = len(is_anomaly)
    
    alert_flags = np.zeros(n, dtype=int)
    cooldown_counter = 0

    for i in range(n):
        if cooldown_counter > 0:
            cooldown_counter -= 1
            continue

        # Kiểm tra điều kiện đủ consecutive điểm bất thường liên tiếp
        if i >= consecutive - 1:
            window = is_anomaly[i - consecutive + 1 : i + 1]
            if np.all(window == 1):
                alert_flags[i] = 1
                cooldown_counter = cooldown  # Kích hoạt cooldown

    df["alert_flag"] = alert_flags
    return df