"""
tests/test_data_pipeline.py
Bộ kiểm thử tự động 6 kịch bản tính toàn vẹn dữ liệu, chống rò rỉ thời gian
và tính tái lập tất định cho pipeline MLOps.
"""
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

def test_temporal_order_and_anti_leakage():
    """Kiểm tra điều kiện chống rò rỉ dữ liệu thời gian: Train kết thúc trước Prediction."""
    train_path = "data/processed/train_clean_normal.parquet"
    pred_path = "data/processed/prediction_clean.parquet"
    
    if os.path.exists(train_path) and os.path.exists(pred_path):
        tr = pd.read_parquet(train_path)
        pr = pd.read_parquet(pred_path)
        t_train_max = pd.to_datetime(tr['time_stamp']).max()
        t_pred_min = pd.to_datetime(pr['time_stamp']).min()
        assert t_train_max <= t_pred_min, f"Rò rỉ dữ liệu: {t_train_max} > {t_pred_min}"
    else:
        pytest.skip("Chưa có dữ liệu Parquet để kiểm tra thời gian.")

def test_scaled_data_integrity():
    """Kiểm tra tập dữ liệu scaled không chứa giá trị NaN hoặc Inf."""
    scaled_path = "data/processed/train_scaled.parquet"
    if os.path.exists(scaled_path):
        df = pd.read_parquet(scaled_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert not df[numeric_cols].isnull().any().any(), "Tập scaled vẫn chứa NaN!"
        assert not np.isinf(df[numeric_cols].values).any(), "Tập scaled chứa giá trị vô cùng (Inf)!"
    else:
        pytest.skip("Chưa có train_scaled.parquet để kiểm tra.")

def test_isolation_forest_reproducibility():
    """Kiểm tra tính tái lập tuyệt đối của Isolation Forest với seed cố định (n_estimators=200)."""
    X = np.random.RandomState(42).randn(100, 5)
    model1 = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    model2 = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    
    score1 = model1.fit(X).decision_function(X)
    score2 = model2.fit(X).decision_function(X)
    
    np.testing.assert_allclose(score1, score2, atol=1e-7, err_msg="Mô hình không đạt tính tất định!")

def test_scaler_mean_and_scale_properties():
    """Kiểm tra Scaler lưu trong models/ có đủ thuộc tính mean_ và scale_."""
    scaler_path = "models/scaler.joblib"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        assert hasattr(scaler, 'mean_'), "Scaler thiếu thuộc tính mean_"
        assert hasattr(scaler, 'scale_'), "Scaler thiếu thuộc tính scale_"
    else:
        pytest.skip("Chưa có models/scaler.joblib.")

def test_defensive_alert_rules_on_extreme_data():
    """Kiểm tra lập trình phòng vệ trước dữ liệu cực đoan: chuỗi NaN và mảng rỗng."""
    from src.alerts.rules import apply_alert_rules
    
    nan_series = pd.Series([np.nan, np.nan, np.nan])
    res_nan = apply_alert_rules(nan_series, threshold=0.0)
    assert (res_nan['alert_triggered'] == 0).all(), "Dữ liệu NaN không được kích hoạt cảnh báo!"

    empty_series = pd.Series([], dtype=float)
    res_empty = apply_alert_rules(empty_series, threshold=0.0)
    assert len(res_empty) == 0, "Dữ liệu rỗng phải trả về DataFrame rỗng!"

def test_dashboard_serving_artifacts():
    """Kiểm tra file phục vụ Dashboard có đủ các cột trọng yếu."""
    scored_path = "data/processed/prediction_scored.parquet"
    if os.path.exists(scored_path):
        df = pd.read_parquet(scored_path)
        required_cols = ['time_stamp', 'anomaly_score', 'iso_label']
        for col in required_cols:
            assert col in df.columns, f"Thiếu cột {col} trong file prediction_scored.parquet!"
    else:
        pytest.skip("Chưa có prediction_scored.parquet.")