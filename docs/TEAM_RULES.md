# Quy ước làm việc nhóm

1. Một người phụ trách một module chính; thống nhất giao diện đầu vào/đầu ra trước khi code.
2. Nhánh Git: `main` (bản ổn định), `dev` (tích hợp), `feat/<ten-viec>` (tính năng).
3. Không commit file dữ liệu gốc, môi trường ảo, file model lớn hoặc file `.env`.
4. Mỗi lần họp phải cập nhật `docs/WORK_LOG.md`.
5. Mỗi kết quả biểu đồ phải có script tái tạo, tiêu đề, đơn vị, khoảng thời gian và ghi chú.
6. Không gọi một điểm bất thường là “hỏng hóc chắc chắn”; dùng “cảnh báo bất thường / cần kiểm tra”.
7. Phải tách theo thời gian hoặc theo cách phân chia train/prediction đã có; không chia ngẫu nhiên làm rò rỉ tương lai.
8. Không dùng dữ liệu tương lai để chuẩn hóa, chọn ngưỡng hoặc huấn luyện.
9. Mọi quyết định loại bỏ dữ liệu, chọn cảm biến, ngưỡng cảnh báo phải ghi vào `docs/DECISIONS.md`.
10. Không sửa cấu trúc thư mục. Nếu cần thêm yêu cầu, điền vào file tương ứng hoặc thống nhất thay đổi phiên bản sau dự án.

# Quy ước phối hợp nhóm & Tiêu chuẩn Repo cá nhân

## 1. Mô hình làm việc

- Trong tuần: Mỗi thành viên làm việc độc lập trên 1 repository GitHub cá nhân theo lộ trình từng ngày.
- Cuối tuần: Cả nhóm họp đối chiếu kết quả (schema dữ liệu, biểu đồ EDA, hiệu năng mô hình), chọn ra giải pháp tối ưu nhất để tích hợp vào repository đại diện của nhóm.

## 2. Quy tắc bắt buộc trên Repo cá nhân

1. Giữ nguyên 100% cấu trúc thư mục `wind_scada_team_template` để đảm bảo khi copy code của nhau cuối tuần không bị lỗi đường dẫn hay lệch import.
2. Không commit file dữ liệu thô (`data/raw/*`), môi trường ảo (`.venv/`), file model lớn (`models/*`) hoặc file `.env` chứa đường dẫn riêng lên GitHub.
3. Chạy mọi lệnh Python từ thư mục gốc dự án.
4. Ghi chép chi tiết mọi lỗi gặp phải trên máy cá nhân vào `docs/WORK_LOG.md` để làm tài liệu chia sẻ trong buổi họp tuần.
