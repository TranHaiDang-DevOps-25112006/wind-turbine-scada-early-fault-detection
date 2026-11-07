# Wind Turbine SCADA Early Fault Detection

Hệ thống phân tích chuỗi thời gian dữ liệu SCADA công nghiệp nhằm phát hiện sớm hư hỏng cơ điện (Early Fault Detection) trên tua-bin gió, xây dựng theo khung chuẩn Big Data & MLOps [Samsung Innovation Campus (SIC) - 2026].

## Mục tiêu

Xây dựng pipeline nghiên cứu phát hiện bất thường sớm từ dữ liệu SCADA tua-bin gió (bộ dữ liệu CARE To Compare - Zenodo 14006163), giảm thiểu tối đa báo động giả (Alarm Fatigue) và cung cấp giao diện Web Dashboard giám sát trực quan hỗ trợ kỹ sư bảo trì.

## Quy tắc làm việc của nhóm

1. Không sửa cấu trúc thư mục này trong suốt dự án; chỉ điền nội dung vào các file đã có.
2. Không đưa dữ liệu gốc dung lượng lớn lên Git (đã chặn qua `.gitignore`).
3. Mỗi thành viên chỉ commit phần việc của mình; không sửa file của người khác khi chưa thống nhất.
4. Chạy lệnh từ thư mục gốc dự án.
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

Chỉ cần **một câu lệnh duy nhất** để tự động thực thi 9 công đoạn liên hoàn: khảo sát dữ liệu, tiền xử lý, đặc trưng, chuẩn hóa, huấn luyện, đánh giá đa ngưỡng (evaluate_advanced), đa sự kiện minh họa (evaluate_multievent), benchmark PySpark, evaluate_real.

```powershell
$env:PYTHONIOENCODING = "utf-8"
python run_pipeline.py
```

### 2. Chạy kiểm thử

```powershell
python -m pytest tests/ -q
```

### 3. Khởi chạy Dashboard

Giao diện đọc kết quả lưu sẵn theo lô (không phải giám sát thời gian thực), mở tại http://localhost:8501:

```powershell
streamlit run src/dashboard/app.py
```

### Dữ liệu

Dữ liệu gốc CARE To Compare (Zenodo record 14006163) và tệp mẫu `data/sample/sample_event_0.csv` **KHÔNG** được đưa lên GitHub vì dung lượng/giấy phép.

**Cách tạo lại mẫu:**

1. Tải `Wind Farm A.zip` từ Zenodo, giải nén.
2. Lấy 3.000 dòng cuối của `0.csv` (id 51986–54985, dấu phân cách `;`) lưu thành `data/sample/sample_event_0.csv`.
3. Chép `event_info.csv` của Wind Farm A vào `data/sample/`.

Pipeline cần hai tệp này để chạy.

### Kết quả tham chiếu (seed 42, Event 0, cấp bản ghi)

| Chỉ số | Giá trị |
|---|---|
| TP | 1 147 |
| FP | 30 |
| TN | 796 |
| FN | 865 |
| Precision | 97,45 % |
| Recall | 57,01 % |
| F1 | 71,93 % |

> **Ghi chú:** Đánh giá trên 1 sự kiện, 3.000 dòng mẫu, 110 dòng train; không phải chẩn đoán hỏng hóc.
