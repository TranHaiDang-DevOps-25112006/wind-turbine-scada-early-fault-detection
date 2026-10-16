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
