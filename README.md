# Wind Turbine SCADA Early Fault Detection

Hệ thống phân tích chuỗi thời gian dữ liệu SCADA công nghiệp nhằm phát hiện sớm hư hỏng cơ điện (Early Fault Detection) trên tua-bin gió, xây dựng theo khung chuẩn Big Data & MLOps [Samsung Innovation Campus (SIC) - 2026].

## Mục tiêu

Xây dựng pipeline nghiên cứu phát hiện bất thường sớm từ dữ liệu SCADA tua-bin gió (bộ dữ liệu CARE To Compare - Zenodo 14006163), giảm thiểu tối đa báo động giả (Alarm Fatigue) và cung cấp giao diện Web Dashboard giám sát trực quan hỗ trợ kỹ sư bảo trì.

## Quy tắc làm việc của nhóm

1. Không sửa cấu trúc thư mục này trong suốt dự án; chỉ điền nội dung vào các file đã có.
2. Không đưa dữ liệu gốc dung lượng lớn lên Git (đã chặn qua `.gitignore`).
3. Mỗi thành viên chỉ commit phần việc của mình; không sửa file của người khác khi chưa thống nhất.
4. Chạy lệnh từ thư mục gốc dự án (`wind_scada_team_template/`).
5. Mọi kết quả cuối cùng phải có thể tái tạo bằng cấu hình và seed cố định (`random_state=42`).

## Cài đặt môi trường

- Yêu cầu Python 3.11 (khuyến nghị 3.11.9) và Git.
- Tạo môi trường ảo: `python -m venv .venv`
- Kích hoạt môi trường ảo:
  - Windows PowerShell: `.venv\Scripts\Activate.ps1`
  - Linux / macOS / Git Bash: `source .venv/bin/activate`
- Cài đặt thư viện: `pip install -r requirements.txt`
- Sao chép `.env.example` thành `.env` nếu cần cấu hình đường dẫn dữ liệu riêng.
- Đọc `docs/DATA_DICTIONARY.md` và `docs/TEAM_RULES.md` trước khi chạy pipeline.

## Hướng dẫn thực thi (Quickstart)

### 1. Chạy liên hoàn toàn bộ Pipeline (End-to-End Runner)

Chỉ cần **một câu lệnh duy nhất** để tự động thực thi 8 công đoạn liên hoàn từ dữ liệu thô, tiền xử lý, trích xuất đặc trưng, huấn luyện Isolation Forest, kiểm thử đơn vị, đến quét đa ngưỡng và đánh giá độ sớm đa sự kiện:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python run_pipeline.py
```

-Khởi chạy giao diện điều hành tương tác thời gian thực trên trình duyệt web (http://localhost:8501):
streamlit run src/dashboard/app.py
