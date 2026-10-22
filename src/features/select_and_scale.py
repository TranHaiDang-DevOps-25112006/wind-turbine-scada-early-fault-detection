import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def run_feature_selection_and_scaling(
    input_dir="data/processed/", 
    output_dir="data/processed/",
    model_dir="models/"
):
    train_path = os.path.join(input_dir, "train_features.parquet")
    pred_path = os.path.join(input_dir, "prediction_features.parquet")

    if not os.path.exists(train_path) or not os.path.exists(pred_path):
        raise FileNotFoundError("[!] Không tìm thấy dữ liệu đặc trưng Ngày 7 trong data/processed/.")

    print("[*] Đang đọc dữ liệu đặc trưng Ngày 7...")
    train_df = pd.read_parquet(train_path)
    pred_df = pd.read_parquet(pred_path)

    # 1. Danh sách các cột Metadata không đưa vào mô hình X
    metadata_cols = ["id", "time_stamp", "asset_id", "train_test", "status_type_id"]
    
    # 2. Danh sách các cột loại bỏ do Đa cộng tuyến hoàn hảo (r = 1.00 từ Ngày 5)
    collinear_cols = ["wind_speed_4_avg"]
    drop_cols = [c for c in (metadata_cols + collinear_cols) if c in train_df.columns]

    # 3. Xác định danh sách feature thô ban đầu
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    print(f" - Tổng số cột ban đầu: {train_df.shape[1]}")
    print(f" - Đã loại bỏ metadata và đa cộng tuyến ({len(drop_cols)} cột): {drop_cols}")
    print(f" - Số lượng đặc trưng đưa vào lựa chọn: {len(feature_cols)}")

    # 4. Lọc bỏ các cột phương sai bằng 0 (nếu có cảm biến hằng số)
    zero_var_cols = [c for c in feature_cols if train_df[c].std() == 0]
    if zero_var_cols:
        print(f" - Loại bỏ các cột phương sai = 0: {zero_var_cols}")
        feature_cols = [c for c in feature_cols if c not in zero_var_cols]

    # Tách ma trận X (chỉ chứa các feature dạng số)
    X_train = train_df[feature_cols].copy()
    X_pred = pred_df[feature_cols].copy()

    # 5. Chuẩn hóa thang đo (StandardScaler) tuân thủ Strict Pipeline Isolation
    print("[*] Đang khởi tạo và khớp StandardScaler trên tập TRAIN...")
    scaler = StandardScaler()
    
    # CHỈ fit trên tập TRAIN
    scaler.fit(X_train)

    # Transform độc lập trên cả 2 tập
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_cols, index=train_df.index)
    X_pred_scaled = pd.DataFrame(scaler.transform(X_pred), columns=feature_cols, index=pred_df.index)

    # Giữ lại các cột định danh để phục vụ vẽ đồ thị và đánh giá sau này
    meta_train = train_df[["time_stamp", "asset_id", "status_type_id"]].copy()
    meta_pred = pred_df[["time_stamp", "asset_id", "status_type_id"]].copy()

    train_scaled_full = pd.concat([meta_train, X_train_scaled], axis=1)
    pred_scaled_full = pd.concat([meta_pred, X_pred_scaled], axis=1)

    # 6. Lưu dữ liệu đã scale và đối tượng scaler
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_out_path = os.path.join(output_dir, "train_scaled.parquet")
    pred_out_path = os.path.join(output_dir, "prediction_scaled.parquet")
    scaler_out_path = os.path.join(model_dir, "scaler.joblib")

    train_scaled_full.to_parquet(train_out_path, index=False)
    pred_scaled_full.to_parquet(pred_out_path, index=False)
    joblib.dump(scaler, scaler_out_path)

    print(f"[✓] Đã xuất tập Train scaled: {train_out_path}")
    print(f"[✓] Đã xuất tập Prediction scaled: {pred_out_path}")
    print(f"[✓] Đã lưu mô hình Scaler: {scaler_out_path}")
    print(f"[*] Kích thước ma trận huấn luyện cuối cùng: {X_train_scaled.shape}")

if __name__ == "__main__":
    run_feature_selection_and_scaling()