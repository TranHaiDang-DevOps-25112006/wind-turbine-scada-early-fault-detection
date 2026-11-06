"""
run_pipeline.py
Kịch bản điều phối thực thi liên hoàn hệ thống SCADA Tua-bin Gió (End-to-End Orchestrator).
Chạy thông suốt 9 bước từ kiểm tra schema đến xuất số liệu thực nghiệm khoa học chuẩn xác.
"""
import sys
import os
import subprocess
import time

ENV_UTF8 = os.environ.copy()
ENV_UTF8['PYTHONIOENCODING'] = 'utf-8'
ENV_UTF8['PYTHONUTF8'] = '1'

STEPS = [
    ("1. Khảo sát dữ liệu thô (Data Inspection)", "src.data.inspect_data"),
    ("2. Tiền xử lý chuỗi thời gian & Làm sạch", "src.data.preprocess"),
    ("3. Trích xuất đặc trưng động lực học", "src.features.build_features"),
    ("4. Chuẩn hóa & Cô lập Pipeline Scaler", "src.features.select_and_scale"),
    ("5. Huấn luyện mô hình Isolation Forest", "src.models.train"),
    ("6. Đánh giá quét đa ngưỡng PR-Curve", "src.evaluation.evaluate_advanced"),
    ("7. Mô phỏng khung kiến trúc đa sự kiện", "src.evaluation.evaluate_multievent"),
    ("8. Đo đạc hiệu năng phân tán PySpark Benchmark", "src.bigdata.spark_benchmark"),
    ("9. Đánh giá thực nghiệm khoa học Event 0 (evaluate_real)", "src.evaluation.evaluate_real")
]

def main():
    print("=" * 75)
    print(" HỆ THỐNG SCADA TUA-BIN GIÓ - QUY TRÌNH THỰC THI LIÊN HOÀN END-TO-END")
    print(" Samsung Innovation Campus (SIC) - 2026")
    print("=" * 75)

    start_total = time.time()

    for idx, (desc, module_name) in enumerate(STEPS, 1):
        print(f"\n[{idx}/{len(STEPS)}] Bắt đầu: {desc}...")
        step_start = time.time()
        
        cmd = [sys.executable, "-m", module_name]
        try:
            res = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=ENV_UTF8
            )
            elapsed = time.time() - step_start
            print(f"[OK] Hoàn thành trong {elapsed:.2f}s")
            if res.stdout:
                last_lines = [line for line in res.stdout.strip().split('\n') if line.strip()]
                if last_lines:
                    print(f"     -> {last_lines[-1]}")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Bước '{desc}' gặp sự cố!")
            print(f"Chi tiết lỗi:\n{e.stderr}")
            sys.exit(1)

    total_elapsed = time.time() - start_total
    print("\n" + "=" * 75)
    print(f"🎉 HOÀN THÀNH TOÀN BỘ 9 BƯỚC PIPELINE THÀNH CÔNG TRONG {total_elapsed:.2f} GIÂY!")
    print("Toàn bộ số liệu thực nghiệm đã sẵn sàng tại: reports/metrics/real_*.csv")
    print("=" * 75)

if __name__ == "__main__":
    main()