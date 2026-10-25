"""
Mục đích: Đóng gói logic quy tắc cảnh báo sự cố, bộ lọc chuỗi liên tiếp và thời gian nghỉ (cooldown)
"""

import numpy as np
import pandas as pd


def apply_alert_rules(
    scores: pd.Series,
    threshold: float = 0.0,
    consecutive_points: int = 3,
    cooldown_points: int = 6,
) -> pd.DataFrame:
  """Áp dụng bộ quy tắc lọc cảnh báo chống quá tải báo động (Alarm Fatigue).

  Tham số:
  ---------
  scores : pd.Series
      Chuỗi điểm bất thường liên tục (anomaly_score = -decision_function).
  threshold : float
      Ngưỡng phân định rủi ro (mặc định: 0.0).
  consecutive_points : int
      Số chu kỳ liên tiếp bắt buộc vượt ngưỡng để kích hoạt còi (3 chu kỳ ~ 30
      phút).
  cooldown_points : int
      Số chu kỳ tạm dừng gửi cảnh báo lặp lại sau khi kích hoạt (6 chu kỳ ~ 1
      giờ).

  Trả về:
  -------
  pd.DataFrame: Bảng gồm 3 cột:
      - 'anomaly_candidate': 1 nếu score >= threshold, ngược lại 0.
      - 'alert_triggered': 1 nếu thỏa mãn luật liên tiếp và cooldown, ngược lại
      0.
      - 'alert_level': Mức độ rủi ro ('NORMAL', 'LOW', 'MEDIUM', 'HIGH').
  """
  # 1. Kiểm tra chuỗi rỗng
  if scores is None or len(scores) == 0:
    return pd.DataFrame(
        columns=['anomaly_candidate', 'alert_triggered', 'alert_level']
    )

  # 2. Xử lý giá trị khuyết thiếu NaN (tránh báo động giả)
  clean_scores = scores.fillna(-999.0)
  is_anomaly = (clean_scores >= threshold).astype(int).values
  n = len(is_anomaly)

  alert_triggered = np.zeros(n, dtype=int)
  consec_count = 0
  cooldown_counter = 0

  # 3. Quét chuỗi thời gian thực thi luật consecutive và cooldown
  for i in range(n):
    if cooldown_counter > 0:
      cooldown_counter -= 1
      consec_count = 0  # Đang trong thời gian nghỉ, reset chuỗi
      continue

    if is_anomaly[i] == 1:
      consec_count += 1
      if consec_count >= consecutive_points:
        alert_triggered[i] = 1
        cooldown_counter = (
            cooldown_points  # Kích hoạt thời gian nghỉ (cooldown)
        )
        consec_count = 0
    else:
      consec_count = 0

  # 4. Phân cấp mức độ nghiêm trọng 
  alert_levels = []
  for s, triggered in zip(clean_scores, alert_triggered):
    if triggered == 1:
      if s >= threshold + 0.04:
        alert_levels.append('HIGH')
      elif s >= threshold + 0.02:
        alert_levels.append('MEDIUM')
      else:
        alert_levels.append('LOW')
    else:
      alert_levels.append('NORMAL')

  return pd.DataFrame(
      {
          'anomaly_candidate': is_anomaly,
          'alert_triggered': alert_triggered,
          'alert_level': alert_levels,
      },
      index=scores.index if isinstance(scores, pd.Series) else None,
  )