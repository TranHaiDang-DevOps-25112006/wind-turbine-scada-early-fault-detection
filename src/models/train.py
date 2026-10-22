import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def train_isolation_forest(
    input_dir="data/processed/", 
    model_dir="models/", 
    output_dir="data/processed/"
):
    train_path = os.path.join(input_dir, "train_scaled.parquet")
    pred_path = os.path.join(input_dir, "prediction_scaled.parquet")

    if not os.path.exists(train_path) or not os.path.exists(pred_path):
        raise FileNotFoundError("[!] Không tìm thấy dữ liệu chuẩn hóa Ngày 8 trong data/processed/.")

    print("[*] Đang nạp dữ liệu đã scale từ Ngày 8...")
    train_df = pd.read_parquet(train_path)
    pred_df = pd.read_parquet(pred_path)

    # 1. Xác định các cột đặc trưng đưa vào huấn luyện X (bỏ qua metadata)
    metadata_cols = ["time_stamp", "asset_id", "status_type_id"]
    feature_cols = [c for c in train_df.columns if c not in metadata_cols]
    
    X_train = train_df[feature_cols]
    X_pred = pred_df[feature_cols]

    print(f" - Số lượng mẫu huấn luyện (Train): {X_train.shape[0]:,} dòng")
    print(f" - Số lượng mẫu kiểm tra (Prediction): {X_pred.shape[0]:,} dòng")
    print(f" - Số lượng đặc trưng đầu vào: {X_train.shape[1]} features")

    # 2. Khởi tạo mô hình Isolation Forest theo cấu hình chuẩn
    print("[*] Bắt đầu huấn luyện Isolation Forest (n_estimators=200, contamination=0.01)...")
    iso_model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )

    # CHỈ huấn luyện trên tập Train bình thường
    iso_model.fit(X_train)
    print("[✓] Huấn luyện hoàn tất!")

    # 3. Tính Anomaly Score & Nhãn cảnh báo thô
    # decision_function: giá trị càng âm càng bất thường -> nhân -1 để score càng cao càng rủi ro
    print("[*] Đang tính toán Anomaly Score trên cả 2 tập...")
    train_df["raw_score"] = iso_model.decision_function(X_train)
    train_df["anomaly_score"] = -train_df["raw_score"]
    train_df["iso_label"] = iso_model.predict(X_train)  # -1: bất thường, 1: bình thường

    pred_df["raw_score"] = iso_model.decision_function(X_pred)
    pred_df["anomaly_score"] = -pred_df["raw_score"]
    pred_df["iso_label"] = iso_model.predict(X_pred)

    # 4. Lưu mô hình đã huấn luyện vào thư mục models/ ở thư mục gốc
    os.makedirs(model_dir, exist_ok=True)
    model_output_path = os.path.join(model_dir, "isolation_forest.joblib")
    joblib.dump(iso_model, model_output_path)
    print(f"[✓] Đã lưu mô hình: {model_output_path}")

    # 5. Lưu tập dữ liệu đã chấm điểm để Ngày 10 đánh giá cảnh báo sớm
    train_scored_path = os.path.join(output_dir, "train_scored.parquet")
    pred_scored_path = os.path.join(output_dir, "prediction_scored.parquet")
    
    train_df.to_parquet(train_scored_path, index=False)
    pred_df.to_parquet(pred_scored_path, index=False)

    print(f"[✓] Đã xuất tập Train Scored: {train_scored_path}")
    print(f"[✓] Đã xuất tập Prediction Scored: {pred_scored_path}")

    # Thống kê nhanh kết quả
    n_train_anomalies = (train_df["iso_label"] == -1).sum()
    n_pred_anomalies = (pred_df["iso_label"] == -1).sum()
    print(f" - Số điểm bất thường phát hiện trên Train: {n_train_anomalies:,} ({n_train_anomalies/len(train_df)*100:.2f}%)")
    print(f" - Số điểm bất thường phát hiện trên Prediction: {n_pred_anomalies:,} ({n_pred_anomalies/len(pred_df)*100:.2f}%)")

if __name__ == "__main__":
    train_isolation_forest()