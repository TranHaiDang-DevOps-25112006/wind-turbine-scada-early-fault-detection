import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

# Thêm đường dẫn gốc dự án
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts.rules import apply_alert_rules

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "sample_event_0.csv"

# --- TEST 1: KIỂM TRA CHỐNG RÒ RỈ THỜI GIAN (DATA LEAKAGE & TEMPORAL ORDER) ---
def test_temporal_order_and_anti_leakage():
    """Đảm bảo thời gian tập Train nghiêm ngặt nằm trước tập Prediction."""
    train_path = PROCESSED_DIR / "train_clean_normal.parquet"
    pred_path = PROCESSED_DIR / "prediction_clean.parquet"
    
    if not (train_path.exists() and pred_path.exists()):
        pytest.skip("Thiếu dữ liệu processed, bỏ qua test.")
        
    df_train = pd.read_parquet(train_path)
    df_pred = pd.read_parquet(pred_path)
    
    t_train_max = pd.to_datetime(df_train['time_stamp']).max()
    t_pred_min = pd.to_datetime(df_pred['time_stamp']).min()
    
    # Tập Train phải kết thúc trước hoặc đúng bằng thời điểm bắt đầu tập Prediction
    assert t_train_max <= t_pred_min, (
        f"Lỗi rò rỉ thời gian! Train Max ({t_train_max}) vượt quá Prediction Min ({t_pred_min})"
    )

# --- TEST 2: KIỂM TRA TÍNH TOÀN VẸN CỦA DỮ LIỆU ĐẦU VÀO (NO NaN / INF) ---
def test_scaled_data_integrity():
    """Xác minh ma trận đầu vào mô hình không chứa NaN hoặc giá trị vô cùng (Inf)."""
    train_scaled_path = PROCESSED_DIR / "train_scaled.parquet"
    pred_scaled_path = PROCESSED_DIR / "prediction_scaled.parquet"
    
    if not (train_scaled_path.exists() and pred_scaled_path.exists()):
        pytest.skip("Thiếu dữ liệu scaled, bỏ qua test.")
        
    df_train_sc = pd.read_parquet(train_scaled_path)
    df_pred_sc = pd.read_parquet(pred_scaled_path)
    
    # Loại các cột metadata
    meta_cols = ['time_stamp', 'asset_id', 'status_type_id', 'id', 'train_test']
    feature_cols = [c for c in df_train_sc.columns if c not in meta_cols]
    
    assert df_train_sc[feature_cols].isna().sum().sum() == 0, "Tập Train Scaled chứa giá trị NaN!"
    assert df_pred_sc[feature_cols].isna().sum().sum() == 0, "Tập Prediction Scaled chứa giá trị NaN!"
    assert not np.isinf(df_train_sc[feature_cols].to_numpy()).any(), "Tập Train Scaled chứa giá trị Inf!"

# --- TEST 3: KIỂM TOÁN TÍNH TÁI LẬP CỦA MÔ HÌNH (MODEL REPRODUCIBILITY) ---
def test_isolation_forest_reproducibility():
    """Huấn luyện 2 lần độc lập với cùng random_state=42 phải cho ra cùng kết quả 100%."""
    train_scaled_path = PROCESSED_DIR / "train_scaled.parquet"
    if not train_scaled_path.exists():
        pytest.skip("Thiếu dữ liệu train_scaled, bỏ qua test.")
        
    df_train_sc = pd.read_parquet(train_scaled_path)
    meta_cols = ['time_stamp', 'asset_id', 'status_type_id', 'id', 'train_test']
    feature_cols = [c for c in df_train_sc.columns if c not in meta_cols]
    X_train = df_train_sc[feature_cols].to_numpy()
    
    # Huấn luyện Run 1
    model_1 = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model_1.fit(X_train)
    scores_1 = -model_1.decision_function(X_train[:20])
    
    # Huấn luyện Run 2
    model_2 = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model_2.fit(X_train)
    scores_2 = -model_2.decision_function(X_train[:20])
    
    # Kiểm tra sai số tuyệt đối xấp xỉ bằng 0
    np.testing.assert_allclose(scores_1, scores_2, atol=1e-7, err_msg="Mô hình không tái lập được kết quả!")

# --- TEST 4: KIỂM TRA TÍNH BẢO VỆ CỦA BỘ CHUẨN HÓA (STRICT SCALER FIT) ---
def test_scaler_mean_and_scale_properties():
    """Bộ Scaler phải được fit trên Train và lưu đúng định dạng."""
    scaler_file = MODELS_DIR / "scaler.joblib"
    assert scaler_file.exists(), "Tệp hiện vật models/scaler.joblib không tồn tại!"
    
    scaler = joblib.load(scaler_file)
    assert hasattr(scaler, "mean_"), "Scaler chưa được fit dữ liệu!"
    assert hasattr(scaler, "scale_"), "Scaler không có thuộc tính phương sai scale_!"
    assert (scaler.scale_ > 0).all(), "Phát hiện biến có độ lệch chuẩn bằng 0 gây lỗi chia cho 0!"

# --- TEST 5: KIỂM THỬ KHẢ NĂNG PHÒNG VỆ CỦA HÀM CẢNH BÁO (DEFENSIVE ALERT RULES) ---
def test_defensive_alert_rules_on_extreme_data():
    """Hàm rules.py phải xử lý an toàn khi gặp dữ liệu lỗi mà không văng Exception."""
    # Ca 1: Chuỗi rỗng
    empty_df = apply_alert_rules(pd.Series([], dtype=float))
    assert len(empty_df) == 0
    assert 'alert_triggered' in empty_df.columns
    
    # Ca 2: Toàn bộ giá trị NaN
    nan_series = pd.Series([np.nan, np.nan, np.nan, np.nan])
    nan_df = apply_alert_rules(nan_series, threshold=0.0)
    assert nan_df['alert_triggered'].sum() == 0, "Dữ liệu NaN bị kích hoạt cảnh báo nhầm!"
    
    # Ca 3: Chuỗi giá trị cực đoan
    extreme_series = pd.Series([999.0, -999.0, 999.0, 999.0, 999.0])
    extreme_df = apply_alert_rules(extreme_series, threshold=0.0, consecutive_points=3)
    assert 'alert_level' in extreme_df.columns

# --- TEST 6: KIỂM TRA ĐỊNH DẠNG ĐẦU RA CHO DASHBOARD ---
def test_dashboard_serving_artifacts():
    """Đảm bảo các file phục vụ Streamlit Dashboard sẵn sàng và đầy đủ cột."""
    scored_path = PROCESSED_DIR / "prediction_scored.parquet"
    if not scored_path.exists():
        pytest.skip("Thiếu tệp prediction_scored.parquet.")
        
    df_scored = pd.read_parquet(scored_path)
    required_cols = ['time_stamp', 'asset_id', 'anomaly_score']
    for col in required_cols:
        assert col in df_scored.columns, f"Thiếu cột bắt buộc '{col}' phục vụ hiển thị Dashboard!"