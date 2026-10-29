import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cưỡng bức encoding UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Wind Farm A" / "datasets" / "0.csv"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "sample_event_0.csv"
REPORTS_METRICS = PROJECT_ROOT / "reports" / "metrics"
REPORTS_FIGURES = PROJECT_ROOT / "reports" / "figures"

REPORTS_METRICS.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)

def benchmark_pandas(file_path: Path):
    print("\n--- [1] Chạy Benchmark với Pandas ---")
    start_total = time.time()
    
    # 1. Đọc dữ liệu
    t0 = time.time()
    df = pd.read_csv(file_path, sep=';')
    t_read = time.time() - t0
    
    # 2. Tiền xử lý & Sắp xếp thời gian
    t0 = time.time()
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df = df.sort_values('time_stamp').reset_index(drop=True)
    df_filtered = df[df['status_type_id'] == 0].copy()
    t_clean = time.time() - t0
    
    # 3. Tạo đặc trưng Rolling Mean 1h (6 chu kỳ 10 phút)
    t0 = time.time()
    df_filtered['rolling_sensor6_1h'] = df_filtered['sensor_6_avg'].rolling(window=6).mean()
    df_filtered['rolling_wind3_1h'] = df_filtered['wind_speed_3_avg'].rolling(window=6).mean()
    t_feature = time.time() - t0
    
    # 4. Gom nhóm tổng hợp
    t0 = time.time()
    df_filtered['date'] = df_filtered['time_stamp'].dt.date
    agg_df = df_filtered.groupby(['asset_id', 'date']).agg({
        'sensor_6_avg': ['mean', 'max'],
        'wind_speed_3_avg': ['mean', 'max']
    })
    t_agg = time.time() - t0
    
    total_time = time.time() - start_total
    row_count = len(df)
    
    print(f"Pandas xử lý {row_count:,} dòng: {total_time:.4f} giây")
    return {
        "engine": "Pandas",
        "rows": row_count,
        "read_sec": t_read,
        "clean_sec": t_clean,
        "feature_sec": t_feature,
        "agg_sec": t_agg,
        "total_sec": total_time
    }

def benchmark_pyspark(file_path: Path):
    print("\n--- [2] Chạy Benchmark với Apache PySpark ---")
    t_init0 = time.time()
    spark = (
        SparkSession.builder
        .appName("WindSCADA_Day16_Benchmark")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    t_init = time.time() - t_init0
    print(f"Khởi tạo SparkSession: {t_init:.2f}s (Cổng Web UI: http://localhost:4040)")

    start_total = time.time()
    
    # 1. Đọc dữ liệu (Schema inference)
    t0 = time.time()
    sdf = spark.read.csv(str(file_path), header=True, sep=';', inferSchema=True)
    sdf = sdf.withColumn("time_stamp", F.to_timestamp("time_stamp"))
    # Ép Spark thực thi Action để đo thời gian đọc thực
    row_count = sdf.count()
    t_read = time.time() - t0
    
    # 2. Tiền xử lý & Lọc trạng thái bình thường
    t0 = time.time()
    sdf_filtered = sdf.filter(F.col("status_type_id") == 0).sort("time_stamp")
    _ = sdf_filtered.count()  # Action
    t_clean = time.time() - t0
    
    # 3. Tạo đặc trưng Window Functions (Rolling 1h = 3600 giây)
    t0 = time.time()
    sdf_filtered = sdf_filtered.withColumn("ts_seconds", F.col("time_stamp").cast("long"))
    window_1h = Window.partitionBy("asset_id").orderBy("ts_seconds").rangeBetween(-3600, 0)
    
    sdf_features = sdf_filtered.withColumn(
        "rolling_sensor6_1h", F.avg("sensor_6_avg").over(window_1h)
    ).withColumn(
        "rolling_wind3_1h", F.avg("wind_speed_3_avg").over(window_1h)
    )
    _ = sdf_features.count()  # Action
    t_feature = time.time() - t0
    
    # 4. Gom nhóm tổng hợp
    t0 = time.time()
    sdf_features = sdf_features.withColumn("date", F.to_date("time_stamp"))
    agg_sdf = sdf_features.groupBy("asset_id", "date").agg(
        F.mean("sensor_6_avg").alias("sensor6_mean"),
        F.max("sensor_6_avg").alias("sensor6_max"),
        F.mean("wind_speed_3_avg").alias("wind3_mean"),
        F.max("wind_speed_3_avg").alias("wind3_max")
    )
    _ = agg_sdf.count()  # Action
    t_agg = time.time() - t0
    
    total_time = time.time() - start_total
    print(f"PySpark xử lý {row_count:,} dòng: {total_time:.4f} giây")
    input("\n>>> Spark UI đang mở tại http://localhost:4040 - nhấn Enter để tắt Spark...")
    
    spark.stop()
    return {
        "engine": "PySpark",
        "rows": row_count,
        "read_sec": t_read,
        "clean_sec": t_clean,
        "feature_sec": t_feature,
        "agg_sec": t_agg,
        "total_sec": total_time
    }

def export_visualizations(df_metrics):
    csv_out = REPORTS_METRICS / "day16_pandas_vs_pyspark_benchmark.csv"
    df_metrics.to_csv(csv_out, index=False, encoding='utf-8')
    print(f"\n[OK] Đã lưu bảng kết quả vào: {csv_out}")
    
    # Vẽ biểu đồ so sánh 4 công đoạn
    stages = ['Đọc dữ liệu\n(Read)', 'Tiền xử lý\n(Clean)', 'Tạo đặc trưng\n(Window)', 'Tổng hợp\n(Agg)']
    pandas_times = [
        df_metrics.loc[df_metrics['engine'] == 'Pandas', 'read_sec'].values[0],
        df_metrics.loc[df_metrics['engine'] == 'Pandas', 'clean_sec'].values[0],
        df_metrics.loc[df_metrics['engine'] == 'Pandas', 'feature_sec'].values[0],
        df_metrics.loc[df_metrics['engine'] == 'Pandas', 'agg_sec'].values[0]
    ]
    spark_times = [
        df_metrics.loc[df_metrics['engine'] == 'PySpark', 'read_sec'].values[0],
        df_metrics.loc[df_metrics['engine'] == 'PySpark', 'clean_sec'].values[0],
        df_metrics.loc[df_metrics['engine'] == 'PySpark', 'feature_sec'].values[0],
        df_metrics.loc[df_metrics['engine'] == 'PySpark', 'agg_sec'].values[0]
    ]
    
    x = np.arange(len(stages))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    rects1 = ax.bar(x - width/2, pandas_times, width, label='Pandas (Single-thread)', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, spark_times, width, label='PySpark (Local Multi-core)', color='#e26d5c')
    
    ax.set_ylabel('Thời gian thực thi (Giây)')
    ax.set_title('So sánh thời gian xử lý chuỗi thời gian SCADA: Pandas vs Apache PySpark', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
                    
    plt.tight_layout()
    fig_out = REPORTS_FIGURES / "day16_pandas_vs_pyspark_benchmark.png"
    plt.savefig(fig_out)
    plt.close()
    print(f"[OK] Đã lưu biểu đồ vào: {fig_out}")

if __name__ == "__main__":
    target_file = DATA_PATH if DATA_PATH.exists() else SAMPLE_PATH
    print(f"[*] Sử dụng tệp: {target_file}")
    
    res_pandas = benchmark_pandas(target_file)
    res_spark = benchmark_pyspark(target_file)
    
    df_metrics = pd.DataFrame([res_pandas, res_spark])
    export_visualizations(df_metrics)
    print("\n=== HOÀN THÀNH BENCHMARK NGÀY 16 ===")