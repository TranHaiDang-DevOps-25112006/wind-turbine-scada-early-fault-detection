"""
Mục đích: Kiểm thử tự động tính đúng đắn và độ bền của logic cảnh báo với pytest
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Thêm đường dẫn gốc dự án vào sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
  sys.path.append(str(ROOT_DIR))

from src.alerts.rules import apply_alert_rules


def test_no_anomalies():
  """Ca 1: Chuỗi hoàn toàn bình thường -> Tuyệt đối không kích hoạt cảnh báo."""
  scores = pd.Series([-0.05, -0.02, -0.01, -0.08, -0.04])
  res = apply_alert_rules(scores, threshold=0.0)

  assert res["alert_triggered"].sum() == 0
  assert (res["alert_level"] == "NORMAL").all()


def test_isolated_outlier_noise():
  """Ca 2: Một điểm xung nhiễu đơn lẻ (gió giật) -> Không kích hoạt cảnh báo."""
  scores = pd.Series([-0.05, 0.06, -0.02, -0.03])
  res = apply_alert_rules(scores, threshold=0.0, consecutive_points=3)

  assert res.loc[1, "anomaly_candidate"] == 1
  assert (
      res["alert_triggered"].sum() == 0
  )  # Bị chặn bởi luật 3 chu kỳ liên tiếp


def test_two_consecutive_points_no_trigger():
  """Ca 3: Mới có 2 điểm liên tiếp (20 phút) -> Chưa đủ điều kiện kích hoạt."""
  scores = pd.Series([-0.05, 0.04, 0.05, -0.02])
  res = apply_alert_rules(scores, threshold=0.0, consecutive_points=3)

  assert res["alert_triggered"].sum() == 0


def test_valid_consecutive_trigger():
  """Ca 4: Đủ 3 chu kỳ liên tiếp (30 phút) -> Kích hoạt đúng tại điểm thứ 3."""
  scores = pd.Series([-0.05, 0.02, 0.03, 0.05, -0.01])
  res = apply_alert_rules(
      scores, threshold=0.0, consecutive_points=3, cooldown_points=3
  )

  assert res.loc[3, "alert_triggered"] == 1
  assert res.loc[3, "alert_level"] == "HIGH"
  assert res["alert_triggered"].sum() == 1


def test_cooldown_suppression():
  """Ca 5: Chuỗi vượt ngưỡng kéo dài liên tục -> Kích hoạt 1 lần, tạm ngắt trong cooldown."""
  # 10 chu kỳ liên tục đều vượt ngưỡng:
  # Điểm 0,1,2 -> Điểm 2 kích hoạt (trigger), cooldown=3 chu kỳ tiếp theo (3,4,5) bị ngắt.
  # Điểm 6,7,8 -> Điểm 8 kích hoạt tiếp lần 2.
  scores = pd.Series([0.05] * 10)
  res = apply_alert_rules(
      scores, threshold=0.0, consecutive_points=3, cooldown_points=3
  )

  assert res.loc[2, "alert_triggered"] == 1
  assert res.loc[3, "alert_triggered"] == 0  # Bị ngắt do cooldown
  assert res.loc[4, "alert_triggered"] == 0
  assert res.loc[5, "alert_triggered"] == 0
  assert res.loc[8, "alert_triggered"] == 1  # Kích hoạt lại sau cooldown
  assert res["alert_triggered"].sum() == 2


def test_empty_and_nan_handling():
  """Ca 6: Dữ liệu rỗng hoặc chứa NaN -> Xử lý trơn tru, không gây sập chương trình."""
  # Chuỗi rỗng
  empty_series = pd.Series([], dtype=float)
  res_empty = apply_alert_rules(empty_series)
  assert len(res_empty) == 0

  # Chuỗi chứa NaN xen kẽ
  nan_series = pd.Series([np.nan, 0.03, np.nan, 0.04, 0.05, 0.06])
  res_nan = apply_alert_rules(nan_series, threshold=0.0, consecutive_points=3)
  assert (
      res_nan.loc[5, "alert_triggered"] == 1
  )  # Điểm 3, 4, 5 đủ 3 chu kỳ liên tiếp
  assert res_nan["alert_triggered"].sum() == 1