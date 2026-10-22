import os
import yaml
import pandas as pd
import numpy as np

def load_config(config_path="configs/project.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_domain_features(df):
    """
    Tạo các đặc trưng chuỗi thời gian miền chuyên sâu cho dữ liệu tua-bin gió.
    Đầu vào: DataFrame đã được sắp xếp thời gian tăng dần.
    """
    df = df.copy()
    
    # 1. Nhóm đặc trưng nhiệt độ tương đối (Relative Temperature)
    # sensor_0_avg là Nhiệt độ môi trường (Ambient Temperature)
    if "sensor_0_avg" in df.columns:
        # Ví dụ sensor_6_avg là nhiệt độ ổ trục/vỏ máy (r = 0.93 với ambient temp)
        if "sensor_6_avg" in df.columns:
            df["sensor_6_delta_ambient"] = df["sensor_6_avg"] - df["sensor_0_avg"]
        if "sensor_7_avg" in df.columns:
            df["sensor_7_delta_ambient"] = df["sensor_7_avg"] - df["sensor_0_avg"]

    # 2. Nhóm đặc trưng Cửa sổ trượt (Rolling Statistics - 1h và 6h)
    # Tần suất 10 phút: 1h = 6 bước, 6h = 36 bước
    target_sensors = [c for c in ["sensor_0_avg", "sensor_6_avg", "sensor_7_avg", "wind_speed_3_avg"] if c in df.columns]
    
    for col in target_sensors:
        # Cửa sổ ngắn hạn 1 giờ (6 quan sát quá khứ)
        df[f"{col}_rolling_mean_1h"] = df[col].rolling(window=6, min_periods=1).mean()
        df[f"{col}_rolling_std_1h"] = df[col].rolling(window=6, min_periods=1).std().fillna(0)
        
        # Cửa sổ trung hạn 6 giờ (36 quan sát quá khứ)
        df[f"{col}_rolling_mean_6h"] = df[col].rolling(window=36, min_periods=1).mean()
        df[f"{col}_rolling_std_6h"] = df[col].rolling(window=36, min_periods=1).std().fillna(0)

    # 3. Nhóm đặc trưng Biến thiên độ trễ (Lag & Delta gia tốc)
    for col in target_sensors:
        # Biến thiên tức thời so với 1 giờ trước (t - 6)
        df[f"{col}_delta_1h"] = df[col] - df[col].shift(6)
        # Biến thiên tức thời so với 10 phút trước (t - 1)
        df[f"{col}_delta_10m"] = df[col] - df[col].shift(1)

    # 4. Nhóm đặc trưng khí động học & phụ tải (Wind-to-Sensor Ratio)
    if "wind_speed_3_avg" in df.columns and "sensor_9_avg" in df.columns:
        # Tránh chia cho 0 khi lặng gió
        df["wind_to_sensor9_ratio"] = df["sensor_9_avg"] / (df["wind_speed_3_avg"] + 1e-5)

    # Xử lý các giá trị NaN sinh ra do shift() ở các dòng đầu tiên
    df = df.bfill()

    return df

def run_feature_engineering(input_dir="data/processed/", output_dir="data/processed/"):
    train_file = os.path.join(input_dir, "train_clean_normal.parquet")
    pred_file = os.path.join(input_dir, "prediction_clean.parquet")

    if not os.path.exists(train_file) or not os.path.exists(pred_file):
        raise FileNotFoundError("[!] Không tìm thấy dữ liệu sạch từ Ngày 6 trong data/processed/.")

    print("[*] Đang đọc dữ liệu sạch...")
    train_df = pd.read_parquet(train_file)
    pred_df = pd.read_parquet(pred_file)

    print(f" - Tạo đặc trưng trên tập Train ({len(train_df):,} dòng)...")
    train_feat = extract_domain_features(train_df)

    print(f" - Tạo đặc trưng trên tập Prediction ({len(pred_df):,} dòng)...")
    pred_feat = extract_domain_features(pred_df)

    # Xuất kết quả ra file parquet hoàn chỉnh đặc trưng
    train_feat_path = os.path.join(output_dir, "train_features.parquet")
    pred_feat_path = os.path.join(output_dir, "prediction_features.parquet")

    train_feat.to_parquet(train_feat_path, index=False)
    pred_feat.to_parquet(pred_feat_path, index=False)

    print(f"[✓] Đã lưu tập Train đầy đủ đặc trưng: {train_feat_path} ({train_feat.shape[1]} cột)")
    print(f"[✓] Đã lưu tập Prediction đầy đủ đặc trưng: {pred_feat_path} ({pred_feat.shape[1]} cột)")

if __name__ == "__main__":
    run_feature_engineering()