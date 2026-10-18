# Nhật ký Quyết định Kỹ thuật (Technical Decisions) - Tuần 1

| Ngày | Quyết định kỹ thuật | Lý do & Căn cứ thực tế | Đề xuất đồng bộ nhóm |
|---|---|---|---|
| **Ngày 1** | Giữ cố định 100% tên thư mục và cấu trúc khung dự án. | Đảm bảo cuối tuần ghép code giữa 5 repo cá nhân không bị xung đột đường dẫn. | Cả 5 thành viên tuyệt đối không tự ý đổi tên thư mục gốc. |
| **Ngày 2** | Chuẩn hóa phiên bản Python 3.11.9 (64-bit) và cô lập qua `.venv`. | Tránh lỗi xung đột thư viện và tương thích hoàn toàn với PySpark trên Windows. | Cả nhóm dùng chung Python 3.11.9 và cài đặt từ `requirements.txt`. |
| **Ngày 3** | Chỉ giải nén trước `Wind Farm A.zip` (~150 MB) để làm mẫu và khám phá schema; giữ nguyên nén Farm B & C. | Tiết kiệm 10-15 GB ổ cứng và tránh treo máy trong giai đoạn xây dựng khung pipeline. | Nhóm thống nhất dùng `Wind Farm A` làm tập chuẩn trong giai đoạn đầu. |
| **Ngày 4** | Cấu hình `csv_sep: ";"`, tách 5 cột metadata (`time_stamp`, `asset_id`, `id`, `train_test`, `status_type_id`) khỏi tập biến cảm biến. | Dữ liệu thực tế dùng dấu `;`; loại bỏ cột định danh và trạng thái khỏi feature giúp ngăn chặn rò rỉ dữ liệu (data leakage). | Đồng bộ `configs/project.yaml` và `docs/DATA_DICTIONARY.md` cho toàn nhóm. |
| **Ngày 5** | Trích xuất 3.000 dòng cuối của `0.csv` vào `data/sample/` và chỉ lọc `status_type_id == 0` khi huấn luyện mô hình bình thường. | Giúp chạy thử code dưới 1 giây; giai đoạn `status_type_id != 0` là lúc tua-bin dừng/bảo trì nên cần lọc riêng khi học phân phối bình thường. | Sử dụng tập mẫu `sample_event_0.csv` để viết unit test và tiền xử lý ở Tuần 2. |
