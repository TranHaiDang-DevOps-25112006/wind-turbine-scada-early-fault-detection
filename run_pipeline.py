"""
Mục đích: Chạy liên hoàn toàn bộ Pipeline End-to-End từ dữ liệu thô đến mô hình và đánh giá
"""

import os
from pathlib import Path
import subprocess
import sys
import time

# Thiết lập thư mục gốc
ROOT_DIR = Path(__file__).resolve().parent

# Ép buộc mã hóa UTF-8 cho tất cả các tiến trình con trên Windows
ENV_UTF8 = os.environ.copy()
ENV_UTF8["PYTHONIOENCODING"] = "utf-8"
ENV_UTF8["PYTHONUTF8"] = "1"

PIPELINE_STEPS = [
    {
        "name": "1. Khảo sát dữ liệu thô (Data Inspection)",
        "module": "src.data.inspect_data",
        "cmd": [sys.executable, "-m", "src.data.inspect_data"],
    },
    {
        "name": "2. Tiền xử lý & Làm sạch chuỗi thời gian (Preprocessing)",
        "module": "src.data.preprocess",
        "cmd": [sys.executable, "-m", "src.data.preprocess"],
    },
    {
        "name": (
            "3. Trích xuất đặc trưng chuỗi thời gian (Feature Engineering)"
        ),
        "module": "src.features.build_features",
        "cmd": [sys.executable, "-m", "src.features.build_features"],
    },
    {
        "name": "4. Lọc biến & Chuẩn hóa thang đo (Select & Scale)",
        "module": "src.features.select_and_scale",
        "cmd": [sys.executable, "-m", "src.features.select_and_scale"],
    },
    {
        "name": "5. Huấn luyện mô hình Isolation Forest (Model Training)",
        "module": "src.models.train",
        "cmd": [sys.executable, "-m", "src.models.train"],
    },
    {
        "name": "6. Kiểm thử tự động bộ lọc cảnh báo (Unit Testing với PyTest)",
        "module": "tests.test_alert_rules",
        "cmd": [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_alert_rules.py",
            "-q",
        ],
    },
    {
        "name": (
            "7. Đánh giá đa ngưỡng & Đối chứng Baseline Z-Score (Evaluation)"
        ),
        "module": "src.evaluation.evaluate_advanced",
        "cmd": [sys.executable, "-m", "src.evaluation.evaluate_advanced"],
    },
    {
        "name": "8. Đánh giá độ sớm cảnh báo đa sự cố (Multi-event Lead Time)",
        "module": "src.evaluation.evaluate_multievent",
        "cmd": [sys.executable, "-m", "src.evaluation.evaluate_multievent"],
    },
]


def main():
  print("=" * 80)
  print(
      "  HỆ THỐNG SCADA TUA-BIN GIÓ - QUY TRÌNH THỰC THI LIÊN HOÀN END-TO-END"
      " (NGÀY 15)"
  )
  print("Samsung Innovation Campus (SIC) - 2026")
  print("=" * 80)
  start_total = time.time()

  for idx, step in enumerate(PIPELINE_STEPS, 1):
    print(f"\n[{idx}/{len(PIPELINE_STEPS)}] Bắt đầu: {step['name']}...")
    t0 = time.time()
    try:
      # Bổ sung encoding='utf-8', errors='replace' và truyền ENV_UTF8
      res = subprocess.run(
          step["cmd"],
          cwd=ROOT_DIR,
          check=True,
          capture_output=True,
          text=True,
          encoding="utf-8",
          errors="replace",
          env=ENV_UTF8,
      )
      elapsed = time.time() - t0
      print(f"      -> [PASSED] Hoàn tất trong {elapsed:.2f}s")
    except subprocess.CalledProcessError as e:
      print(f"\n[ERROR] Bước '{step['name']}' gặp sự cố!")
      print(e.stderr if e.stderr else e.stdout)
      sys.exit(1)

  total_time = time.time() - start_total
  print("\n" + "=" * 80)
  print("  [TỔNG KẾT] Toàn bộ 8 công đoạn Pipeline chạy thành công 100%!")
  print(f"  Thời gian thực thi toàn diện: {total_time:.2f} giây")
  print(
      "  Hiện vật sẵn sàng tại: models/, reports/metrics/, reports/figures/"
  )
  print(
      "  Khởi chạy Dashboard bằng lệnh: streamlit run src/dashboard/app.py"
  )
  print("=" * 80)


if __name__ == "__main__":
  main()