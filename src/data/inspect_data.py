import os
import yaml
import pandas as pd

def load_config(config_path="configs/project.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def inspect_dataset(file_path, sep=";"):
    """
    Kiểm tra cấu trúc và tính toàn vẹn của tệp SCADA.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy tệp: {file_path}")

    print(f"[*] Bắt đầu kiểm tra tệp: {file_path}")
    df = pd.read_csv(file_path, sep=sep)
    
    # 1. Kiểm tra kích thước
    n_rows, n_cols = df.shape
    print(f" - Kích thước: {n_rows:,} dòng, {n_cols} cột")
    
    # 2. Kiểm tra cột thời gian
    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])
        t_min = df["time_stamp"].min()
        t_max = df["time_stamp"].max()
        print(f" - Phạm vi thời gian: từ {t_min} đến {t_max}")
        
        # Kiểm tra thứ tự thời gian
        is_sorted = df["time_stamp"].is_monotonic_increasing
        print(f" - Thứ tự thời gian tăng dần: {is_sorted}")
    else:
        print(" [!] Cảnh báo: Không tìm thấy cột 'time_stamp'")

    # 3. Kiểm tra trùng lặp khóa logic (asset_id + time_stamp)
    if "asset_id" in df.columns and "time_stamp" in df.columns:
        dup_count = df.duplicated(subset=["asset_id", "time_stamp"]).sum()
        print(f" - Số dòng trùng lặp (asset_id, time_stamp): {dup_count}")

    # 4. Kiểm tra phân bố tập dữ liệu và trạng thái (đã sửa lỗi to_string)
    if "train_test" in df.columns:
        print(" - Phân bố train_test:")
        for val, count in df["train_test"].value_counts(dropna=False).items():
            print(f"     + {val}: {count:,} dòng")
        
    if "status_type_id" in df.columns:
        print(" - Phân bố status_type_id:")
        for val, count in df["status_type_id"].value_counts(dropna=False).items():
            print(f"     + Trạng thái {val}: {count:,} dòng")

    # 5. Kiểm tra giá trị khuyết thiếu tổng thể
    total_missing = df.isnull().sum().sum()
    cols_with_missing = (df.isnull().sum() > 0).sum()
    print(f" - Tổng giá trị NaN: {total_missing:,} (trên {cols_with_missing} cột)")

    return df

if __name__ == "__main__":
    config = load_config()
    sample_path = "data/sample/sample_event_0.csv"
    
    if os.path.exists(sample_path):
        inspect_dataset(sample_path, sep=config.get("csv_sep", ";"))
    else:
        print(f"[!] Chưa có tập sample tại {sample_path}. Vui lòng kiểm tra lại Ngày 5.")