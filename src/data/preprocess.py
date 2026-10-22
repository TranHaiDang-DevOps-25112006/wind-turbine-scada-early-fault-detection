import os
import yaml
import pandas as pd
import numpy as np

def load_config(config_path="configs/project.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def clean_and_impute_timeseries(df, limit_steps=3):
    """
    Sắp xếp thời gian, xử lý trùng lặp và điền khuyết ngắn hạn.
    """
    df = df.copy()
    
    # 1. Ép kiểu thời gian và sắp xếp chuỗi
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df = df.sort_values(by=["asset_id", "time_stamp"]).reset_index(drop=True)
    
    # 2. Loại bỏ bản ghi trùng lặp hoàn toàn theo khóa
    initial_len = len(df)
    df = df.drop_duplicates(subset=["asset_id", "time_stamp"], keep="first")
    dropped_dups = initial_len - len(df)
    if dropped_dups > 0:
        print(f"   [+] Đã loại bỏ {dropped_dups} dòng trùng lặp khóa.")

    # 3. Xác định các cột cảm biến đo đạc (loại trừ metadata)
    metadata_cols = ["id", "time_stamp", "asset_id", "train_test", "status_type_id"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    # 4. Điền khuyết ngắn hạn bằng Forward Fill (tối đa 3 bước = 30 phút)
    df[feature_cols] = df[feature_cols].ffill(limit=limit_steps)
    
    # Điền bù các điểm đầu chuỗi bằng Backward Fill (tối đa 1 bước nếu có)
    df[feature_cols] = df[feature_cols].bfill(limit=1)

    return df

def run_preprocessing(input_path, output_dir, config):
    sep = config.get("csv_sep", ";")
    print(f"[*] Đang xử lý dữ liệu từ: {input_path}")
    raw_df = pd.read_csv(input_path, sep=sep)
    
    # 1. Làm sạch thứ tự và điền khuyết
    cleaned_df = clean_and_impute_timeseries(raw_df, limit_steps=3)
    
    # 2. Tách tập train và prediction theo thiết kế
    train_df = cleaned_df[cleaned_df["train_test"] == "train"].copy()
    pred_df = cleaned_df[cleaned_df["train_test"] == "prediction"].copy()

    # 3. Lọc bỏ trạng thái dừng máy trên tập TRAIN (chỉ giữ status_type_id == 0)
    train_normal_df = train_df[train_df["status_type_id"] == 0].copy()
    print(f" - Train ban đầu: {len(train_df):,} dòng -> Sau khi lọc normal (status=0): {len(train_normal_df):,} dòng")
    print(f" - Prediction giữ nguyên: {len(pred_df):,} dòng")

    # 4. Ghi kết quả ra thư mục processed dưới dạng Parquet
    os.makedirs(output_dir, exist_ok=True)
    train_output = os.path.join(output_dir, "train_clean_normal.parquet")
    pred_output = os.path.join(output_dir, "prediction_clean.parquet")
    
    train_normal_df.to_parquet(train_output, index=False)
    pred_df.to_parquet(pred_output, index=False)
    
    print(f"[✓] Đã xuất tập Train sạch: {train_output}")
    print(f"[✓] Đã xuất tập Prediction sạch: {pred_output}")

if __name__ == "__main__":
    cfg = load_config()
    sample_file = "data/sample/sample_event_0.csv"
    processed_dir = "data/processed/"
    
    if os.path.exists(sample_file):
        run_preprocessing(sample_file, processed_dir, cfg)
    else:
        print(f"[!] Không tìm thấy {sample_file}. Cần tạo tập sample trước khi chạy.")