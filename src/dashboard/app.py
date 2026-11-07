"""
src/dashboard/app.py
Giao diện Giám sát & Cảnh báo Sớm Bất thường Tua-bin Gió từ Dữ liệu SCADA
Triển khai bằng Streamlit & Plotly theo kiến trúc Offline Batch Serving.
"""
import os
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path để nhận diện package src khi chạy Streamlit
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.alerts.rules import apply_alert_rules

st.set_page_config(
    page_title="Giám Sát SCADA Tua-bin Gió",
    page_icon="🌬️",
    layout="wide"
)

PROCESSED_FILE = ROOT_DIR / "data" / "processed" / "prediction_scored.parquet"

@st.cache_data
def load_event0_data():
    if not PROCESSED_FILE.exists():
        return None
    df = pd.read_parquet(PROCESSED_FILE)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    return df

def generate_mock_event(event_id):
    """Mô phỏng dữ liệu giao diện cho các sự cố mở rộng.
    Dữ liệu mô phỏng, không phải kết quả mô hình trên dữ liệu thật.
    """
    date_rng = pd.date_range(start="2023-05-01", periods=1500, freq='10min')
    np.random.seed(event_id)
    base_score = np.random.normal(loc=-0.05, scale=0.03, size=len(date_rng))
    base_score[500:1100] += np.random.normal(loc=0.08, scale=0.02, size=600)
    df_mock = pd.DataFrame({
        'time_stamp': date_rng,
        'asset_id': event_id,
        'anomaly_score': base_score,
        'status_type_id': 0
    })
    return df_mock

# SIDEBAR: CẤU HÌNH BỘ LỌC
st.sidebar.title("🌬️ Điều Khiển Giám Sát")
st.sidebar.markdown("**Dự án Big Data & MLOps SCADA**")

event_options = {
    0: "Event 0 - Generator Bearing (Dữ liệu SCADA thực tế)",
    3: "Event 3 - Gearbox (Giao diện thử nghiệm mở rộng)",
    14: "Event 14 - Hydraulic (Giao diện thử nghiệm mở rộng)",
    17: "Event 17 - Transformer (Giao diện thử nghiệm mở rộng)"
}

selected_event = st.sidebar.selectbox(
    "🎯 Chọn Tua-bin / Dạng sự cố:",
    options=list(event_options.keys()),
    format_func=lambda x: event_options[x]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Cấu Hình Bộ Lọc Cảnh Báo")
threshold_val = st.sidebar.slider("Ngưỡng rủi ro (Threshold):", min_value=-0.05, max_value=0.05, value=0.00, step=0.01)
consecutive_pts = st.sidebar.number_input("Số chu kỳ liên tiếp (3 kỳ = 30p):", min_value=1, max_value=6, value=3)
cooldown_pts = st.sidebar.number_input("Thời gian nghỉ sau cảnh báo (6 kỳ = 1h):", min_value=1, max_value=12, value=6)

st.sidebar.info("💡 **Ghi chú vận hành:** Luật 3 chu kỳ liên tiếp + cooldown 6 bản ghi dùng để giảm cảnh báo lặp; mức giảm chưa được định lượng.")

# LOAD DỮ LIỆU
if selected_event == 0:
    df_data = load_event0_data()
    window_start = pd.Timestamp("2023-08-05 06:10:00")
    window_end = pd.Timestamp("2023-08-19 06:10:00")
    component_name = "Vòng bi máy phát (Generator Bearing)"
else:
    df_data = generate_mock_event(selected_event)
    window_start = df_data['time_stamp'].iloc[500]
    window_end = df_data['time_stamp'].iloc[1100]
    component_name = "Thành phần phụ trợ (Mô phỏng)"

# TIÊU ĐỀ TRANG
st.title("🌬️ HỆ THỐNG GIÁM SÁT & PHÁT HIỆN SỚM SỰ CỐ TUA-BIN GIÓ")
st.markdown("**Trang trại điện gió Wind Farm A** | Bộ dữ liệu SCADA công nghiệp *CARE To Compare*")

if selected_event != 0:
    st.warning("⚠️ **Lưu ý minh bạch học thuật:** Dữ liệu chuỗi thời gian của sự cố này đang ở chế độ mô phỏng giao diện (Mockup Data) nhằm kiểm thử khả năng mở rộng đa thiết bị của hệ thống.")

if df_data is None:
    st.error(f"Không tìm thấy dữ liệu `{PROCESSED_FILE}`. Vui lòng chạy pipeline huấn luyện trước!")
    st.stop()

# ÁP DỤNG BỘ LỌC CẢNH BÁO
alert_res = apply_alert_rules(
    df_data['anomaly_score'], 
    threshold=threshold_val, 
    consecutive_points=consecutive_pts, 
    cooldown_points=cooldown_pts
)
df_data['alert_triggered'] = alert_res['alert_triggered']
df_data['alert_level'] = alert_res['alert_level']

# TÍNH TOÁN KPI
total_alerts = int(df_data['alert_triggered'].sum())
high_alerts = int((df_data['alert_level'] == 'HIGH').sum())

alerts_in_window = df_data[(df_data['alert_triggered'] == 1) & (df_data['time_stamp'] >= window_start) & (df_data['time_stamp'] <= window_end)]
if len(alerts_in_window) > 0:
    first_alert = alerts_in_window['time_stamp'].min()
    lead_time_days = (window_end - first_alert).total_seconds() / 86400.0
    delay_days = (first_alert - window_start).total_seconds() / 86400.0
else:
    lead_time_days = 0.0
    delay_days = 0.0

# HIỂN THỊ KPI CARDS
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("📍 Tua-bin Giám Sát", f"WTG_{selected_event:02d}")
kpi2.metric("⏱️ Lead Time Dừng Máy", f"{lead_time_days:.2f} Ngày")
# Đọc Precision từ file seed stability nếu có
_precision_display = "97.45% (cấp bản ghi, seed 42, giá trị lưu sẵn)"
try:
    _seed_csv_path = ROOT_DIR / "reports" / "metrics" / "real_seed_stability.csv"
    if _seed_csv_path.exists():
        _seed_df = pd.read_csv(_seed_csv_path)
        _row42 = _seed_df[_seed_df["seed"] == 42]
        if len(_row42) > 0:
            _raw_prec = float(_row42.iloc[0]["raw_precision"])
            _precision_display = f"{_raw_prec * 100:.2f}% (cấp bản ghi, seed 42)"
except Exception:
    pass  # Giữ giá trị lưu sẵn, dashboard không crash

kpi3.metric("🎯 Độ Chính Xác (Precision)", _precision_display if selected_event == 0 else "N/A")
kpi4.metric("🚨 Tổng Cảnh Báo", f"{total_alerts} Lần")
kpi5.metric("🔥 Cảnh Báo Cấp HIGH", f"{high_alerts} Điểm", delta=f"{delay_days:.2f}d trễ" if delay_days > 0 else "0d")

st.markdown("---")
st.subheader("📈 Phân Tích Chuỗi Thời Gian Anomaly Score & Cửa Sổ Sự Cố (Plotly Interactive)")

# VẼ BIỂU ĐỒ PLOTLY
fig = go.Figure()

# Đường Anomaly Score
fig.add_trace(go.Scatter(
    x=df_data['time_stamp'],
    y=df_data['anomaly_score'],
    mode='lines',
    name='Anomaly Score',
    line=dict(color='#1f77b4', width=1.5)
))

# Đường Ngưỡng rủi ro
fig.add_trace(go.Scatter(
    x=df_data['time_stamp'],
    y=[threshold_val] * len(df_data),
    mode='lines',
    name=f'Ngưỡng rủi ro (thr = {threshold_val:.2f})',
    line=dict(color='gray', width=1, dash='dash')
))

# Vùng sự cố Ground Truth
fig.add_vrect(
    x0=window_start, x1=window_end,
    fillcolor="rgba(255, 0, 0, 0.15)",
    layer="below", line_width=0,
    annotation_text=f"Anomaly Window ({component_name})",
    annotation_position="top left"
)

# Điểm cảnh báo được kích hoạt
alert_points = df_data[df_data['alert_triggered'] == 1]
fig.add_trace(go.Scatter(
    x=alert_points['time_stamp'],
    y=alert_points['anomaly_score'],
    mode='markers',
    name='Cảnh Báo Kích Hoạt (3 chu kỳ)',
    marker=dict(color='#d62728', symbol='triangle-up', size=9),
    customdata=alert_points[['alert_level']],
    hovertemplate="<b>Thời gian:</b> %{x}<br><b>Score:</b> %{y:.4f}<br><b>Mức:</b> %{customdata[0]}<extra></extra>"
))

fig.update_layout(
    height=450,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Thời gian vận hành (time_stamp)",
    yaxis_title="Anomaly Score (-decision_function)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# BẢNG ƯU TIÊN KIỂM TRA
st.subheader("📋 Danh Sách Cảnh Báo Cần Ưu Tiên Kiểm Tra")
if len(alert_points) > 0:
    table_df = alert_points[['time_stamp', 'asset_id', 'anomaly_score', 'alert_level']].copy()
    
    def assign_action(level):
        if level == 'HIGH':
            return "🔴 DỪNG MÁY / KIỂM TRA KHẨN CẤP (Tàu cẩu)"
        elif level == 'MEDIUM':
            return "🟡 Theo dõi sát sao & Giảm tải công suất"
        else:
            return "🟢 Tiếp tục theo dõi định kỳ"
            
    table_df['Khuyến nghị vận hành'] = table_df['alert_level'].apply(assign_action)
    st.dataframe(table_df.sort_values(by='time_stamp', ascending=True), use_container_width=True)
else:
    st.success("✅ Không có cảnh báo nguy cơ nào vượt ngưỡng trong khoảng thời gian này.")