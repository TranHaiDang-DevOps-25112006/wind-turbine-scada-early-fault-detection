# Wind Turbine SCADA Early Fault Detection

## Mục tiêu
Xây dựng pipeline nghiên cứu phát hiện bất thường sớm từ dữ liệu SCADA tua-bin gió và giao diện cảnh báo minh họa.

## Quy tắc làm việc của nhóm
1. Không sửa cấu trúc thư mục này trong suốt dự án; chỉ điền nội dung vào các file đã có.
2. Không đưa dữ liệu gốc dung lượng lớn lên Git.
3. Mỗi thành viên chỉ commit phần việc của mình; không sửa file của người khác khi chưa thống nhất.
4. Chạy lệnh từ thư mục gốc dự án.
5. Mọi kết quả cuối cùng phải có thể tái tạo bằng cấu hình và seed cố định.

## Bắt đầu
- Cài Python 3.11 (khuyến nghị) và Git.
- Tạo môi trường ảo: `python -m venv .venv`
- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- Cài thư viện: `pip install -r requirements.txt`
- Sao chép `.env.example` thành `.env` nếu cần cấu hình đường dẫn dữ liệu.
- Đọc `docs/DATA_DICTIONARY.md` và `docs/TEAM_RULES.md` trước khi chạy pipeline.

## Thứ tự chạy dự kiến
1. Kiểm tra dữ liệu: `python -m src.data.inspect_data`
2. Tiền xử lý: `python -m src.data.preprocess`
3. Tạo đặc trưng: `python -m src.features.build_features`
4. Huấn luyện: `python -m src.models.train`
5. Đánh giá: `python -m src.evaluation.evaluate`
6. Chạy dashboard: `streamlit run src/dashboard/app.py`

Các file Python trên là khung cố định để nhóm điền code; nếu chưa được triển khai thì lệnh tương ứng chưa chạy được. Không tự ý tạo file mới: hãy điền vào đúng file đã có.
