"""
src/bigdata/spark_benchmark.py
Kịch bản đo đạc và đối chứng hiệu năng xử lý chuỗi thời gian giữa Pandas và Apache PySpark.

Ghi chú:
- Cửa sổ rolling của Pandas và Spark đều theo số hàng (6 bản ghi).
- Thời gian Spark ở chế độ local[*], không phải cụm phân tán.
- Không dùng cache().
"""
import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

SAMPLE_PATH = ROOT_DIR / "data" / "sample" / "sample_event_0.csv"
METRICS_DIR = ROOT_DIR / "reports" / "metrics"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

def benchmark_pandas(file_path):
    t0 = time.time()
    df = pd.read_csv(file_path, sep=';')
    t_read = time.time() - t0
    
    t1 = time.time()
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df_clean = df[df['status_type_id'] == 0].sort_values('time_stamp')
    t_clean = time.time() - t1
    
    t2 = time.time()
    df_clean['sensor_6_rolling_1h'] = df_clean['sensor_6_avg'].rolling(window=6, min_periods=1).mean()
    t_feat = time.time() - t2
    
    t3 = time.time()
    df_clean['date'] = df_clean['time_stamp'].dt.date
    agg_df = df_clean.groupby(['asset_id', 'date'])['sensor_6_avg'].mean().reset_index()
    t_agg = time.time() - t3
    
    total = time.time() - t0
    return {"engine": "Pandas", "rows": len(df), "init_sec": 0.0,
            "read_sec": round(t_read, 6), 
            "clean_sec": round(t_clean, 6), "feature_sec": round(t_feat, 6), 
            "agg_sec": round(t_agg, 6), "total_sec": round(total, 6)}

def benchmark_pyspark(file_path):
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    from pyspark.sql.window import Window
    
    t0 = time.time()
    spark = SparkSession.builder \
        .appName("WindSCADA_Benchmark") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    init_sec = time.time() - t0
    
    # Đặt lại t0 để đo riêng thời gian đọc CSV (không tính khởi tạo Spark)
    t0 = time.time()
    
    # 1. Đọc dữ liệu & lưu trước số dòng n_rows
    df_sp = spark.read.csv(str(file_path), header=True, sep=';')
    n_rows = df_sp.count()
    t_read = time.time() - t0
    
    # 2. Tiền xử lý
    t1 = time.time()
    df_clean = df_sp.filter(F.col("status_type_id") == 0) \
                    .withColumn("time_stamp", F.to_timestamp("time_stamp")) \
                    .withColumn("sensor_6_avg", F.col("sensor_6_avg").cast("double"))
    _ = df_clean.count()
    t_clean = time.time() - t1
    
    # 3. Tạo đặc trưng Rolling window
    t2 = time.time()
    w = Window.partitionBy("asset_id").orderBy("time_stamp").rowsBetween(-5, 0)
    df_feat = df_clean.withColumn("sensor_6_rolling_1h", F.avg("sensor_6_avg").over(w))
    _ = df_feat.count()
    t_feat = time.time() - t2
    
    # 4. Tổng hợp
    t3 = time.time()
    df_agg = df_feat.withColumn("date", F.to_date("time_stamp")) \
                    .groupBy("asset_id", "date") \
                    .agg(F.avg("sensor_6_avg").alias("daily_sensor6_avg"))
    _ = df_agg.count()
    t_agg = time.time() - t3
    
    total = init_sec + (time.time() - t0)
    
    # Đóng SparkSession an toàn sau khi đã hoàn tất các phép tính
    spark.stop()
    
    # Dùng biến n_rows đã lưu sẵn, không gọi df_sp.count() lại khi Spark đã dừng
    return {"engine": "PySpark", "rows": n_rows, "init_sec": round(init_sec, 6),
            "read_sec": round(t_read, 6), 
            "clean_sec": round(t_clean, 6), "feature_sec": round(t_feat, 6), 
            "agg_sec": round(t_agg, 6), "total_sec": round(total, 6)}

def main():
    if not SAMPLE_PATH.exists():
        print(f"[ERROR] Không tìm thấy {SAMPLE_PATH}")
        return

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] Đang chạy Benchmark Pandas...")
    res_pandas = benchmark_pandas(SAMPLE_PATH)

    print("[*] Đang chạy Benchmark PySpark...")
    res_spark = benchmark_pyspark(SAMPLE_PATH)

    df_bench = pd.DataFrame([res_pandas, res_spark])
    out_csv = METRICS_DIR / "day16_pandas_vs_pyspark_benchmark.csv"
    df_bench.to_csv(out_csv, index=False)
    print(f"[OK] Đã hoàn thành Benchmark PySpark: {out_csv}")

if __name__ == "__main__":
    main()