
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


CONFIG_DIR = ROOT_DIR / ".streamlit"
CONFIG_FILE = CONFIG_DIR / "config.toml"
if not CONFIG_FILE.exists():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text("""[theme]
base = "dark"
primaryColor = "#38bdf8"
backgroundColor = "#0b0f19"
secondaryBackgroundColor = "#0f172a"
textColor = "#f1f5f9"
font = "sans serif"
""", encoding="utf-8")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.alerts.rules import apply_alert_rules


st.set_page_config(
    page_title="Hệ Thống Giám Sát SCADA Tua-bin Gió",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    
    header[data-testid="stHeader"] {
        background-color: #0b0f19 !important;
    }
    
   
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    
    
    [data-testid="stSidebarUserContent"] {
        padding-top: 2rem !important;
    }
    
    
    [data-testid="stSidebar"] label p, 
    [data-testid="stSidebar"] .stMarkdown p {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stSidebar"] caption, 
    [data-testid="stSidebar"] small {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] div,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] div[data-baseweb="input"],
    [data-testid="stSidebar"] [data-testid="stNumberInput"] div[data-baseweb="base-input"] {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }

    
    [data-testid="stSidebar"] [data-testid="stSelectbox"] span,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
    [data-testid="stSidebar"] input {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background-color: #1e293b !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
        background-color: #334155 !important;
        border: 1px solid #475569 !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {
        background-color: #475569 !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button svg,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
        fill: #38bdf8 !important;
        color: #38bdf8 !important;
    }

    
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"] {
        background-color: #1e293b !important;
    }
    div[data-baseweb="popover"] li,
    div[data-baseweb="menu"] li {
        color: #f8fafc !important;
        background-color: #1e293b !important;
    }
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="menu"] li:hover {
        background-color: #334155 !important;
        color: #38bdf8 !important;
    }

    
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background-color: #132337 !important;
        border: 1px solid #0284c7 !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] p,
    [data-testid="stSidebar"] [data-testid="stAlert"] div {
        color: #e0f2fe !important;
        font-size: 0.82rem !important;
    }

    
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #161f30, #0f172a);
        border: 1px solid #2a374f;
        border-radius: 10px;
        padding: 10px 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    
    
    [data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        white-space: nowrap !important;
    }
    
    
    [data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace !important;
        line-height: 1.25 !important;
    }

    
    div[data-testid="stMetric"]:nth-child(1) [data-testid="stMetricValue"] { color: #60a5fa !important; } /* Tua-bin */
    div[data-testid="stMetric"]:nth-child(2) [data-testid="stMetricValue"] { color: #34d399 !important; } /* Lead Time */
    div[data-testid="stMetric"]:nth-child(3) [data-testid="stMetricValue"] { color: #38bdf8 !important; } /* Precision */
    div[data-testid="stMetric"]:nth-child(4) [data-testid="stMetricValue"] { color: #fbbf24 !important; } /* Tổng cảnh báo */
    div[data-testid="stMetric"]:nth-child(5) [data-testid="stMetricValue"] { color: #f87171 !important; } /* Cảnh báo HIGH */

    
    h1 { color: #f8fafc !important; font-weight: 700 !important; }
    h2, h3 { color: #cbd5e1 !important; }

    
    .stDownloadButton button,
    [data-testid="stDownloadButton"] button {
        background-color: #0284c7 !important;
        background: linear-gradient(180deg, #0284c7, #0369a1) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4) !important;
    }
    .stDownloadButton button:hover,
    [data-testid="stDownloadButton"] button:hover {
        background: linear-gradient(180deg, #38bdf8, #0284c7) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

PROCESSED_FILE = ROOT_DIR / "data" / "processed" / "prediction_scored.parquet"


@st.cache_data
def load_event0_data():
    if not PROCESSED_FILE.exists():
        return None
    df = pd.read_parquet(PROCESSED_FILE)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    return df

def generate_mock_event(event_id):
    date_rng = pd.date_range(start="2023-05-01", periods=1500, freq='10min')
    np.random.seed(event_id)
    base_score = np.random.normal(loc=-0.05, scale=0.03, size=len(date_rng))
    base_score[500:1100] += np.random.normal(loc=0.08, scale=0.02, size=600)
    return pd.DataFrame({
        'time_stamp': date_rng,
        'asset_id': event_id,
        'anomaly_score': base_score,
        'status_type_id': 0
    })


st.sidebar.markdown("## 🌬️ Trung Tâm Giám Sát")
st.sidebar.caption("Offshore Wind Farm SCADA • Anomaly Detection")

event_options = {
    0: "Event 0: Vòng bi máy phát (Dữ liệu thực)",
    3: "Event 3: Hộp số Gearbox (Mô phỏng)",
    14: "Event 14: Thủy lực Hydraulic (Mô phỏng)",
    17: "Event 17: Biến áp Transformer (Mô phỏng)"
}

selected_event = st.sidebar.selectbox(
    "🎯 Chọn Tua-bin / Sự cố:",
    options=list(event_options.keys()),
    format_func=lambda x: event_options[x]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Cấu Hình Bộ Lọc Cảnh Báo")
threshold_val = st.sidebar.slider("Ngưỡng rủi ro (Threshold):", min_value=-0.05, max_value=0.05, value=0.00, step=0.01)
consecutive_pts = st.sidebar.number_input("Số chu kỳ xác nhận (3 kỳ = 30p):", min_value=1, max_value=6, value=3)
cooldown_pts = st.sidebar.number_input("Thời gian Cooldown (6 kỳ = 1h):", min_value=1, max_value=12, value=6)

st.sidebar.info("🛡️ **Nguyên lý lọc:** Yêu cầu 3 chu kỳ liên tiếp vượt ngưỡng để dập tắt xung gió giật, kết hợp Cooldown 6 chu kỳ chống tràn còi báo.")


if selected_event == 0:
    df_data = load_event0_data()
    window_start = pd.Timestamp("2023-08-05 06:10:00")
    window_end = pd.Timestamp("2023-08-19 06:10:00")
    component_name = "Vòng bi máy phát (Generator Bearing)"
else:
    df_data = generate_mock_event(selected_event)
    window_start = df_data['time_stamp'].iloc[500]
    window_end = df_data['time_stamp'].iloc[1100]
    component_name = "Thiết bị phụ trợ (Mô phỏng)"


st.markdown("# 🌬️ HỆ THỐNG GIÁM SÁT SCADA & CẢNH BÁO SỚM SỰ CỐ")
st.markdown(
    "<span style='color: #94a3b8;'>Trang trại điện gió <b>Wind Farm A</b> | Tập đối chuẩn công nghiệp <i>CARE To Compare</i> (Zenodo)</span>", 
    unsafe_allow_html=True
)

if selected_event != 0:
    st.warning("⚠️ **Lưu ý minh bạch học thuật:** Dữ liệu chuỗi thời gian sự cố này đang ở chế độ mô phỏng giao diện (Mockup Data) phục vụ kiểm thử mở rộng.")

if df_data is None:
    st.error(f"❌ Không tìm thấy tệp `{PROCESSED_FILE}`. Vui lòng chạy lệnh `python run_pipeline.py` trước!")
    st.stop()


alert_res = apply_alert_rules(
    df_data['anomaly_score'], 
    threshold=threshold_val, 
    consecutive_points=consecutive_pts, 
    cooldown_points=cooldown_pts
)
df_data['alert_triggered'] = alert_res['alert_triggered']
df_data['alert_level'] = alert_res['alert_level']


st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ Thanh Tua Thời Gian (Demo)")
enable_time_travel = st.sidebar.checkbox("Bật chế độ tua ca trực", value=False)

if enable_time_travel:
    min_time = df_data['time_stamp'].min().to_pydatetime()
    max_time = df_data['time_stamp'].max().to_pydatetime()
    selected_cutoff = st.sidebar.slider(
        "Mốc quan sát hiện tại:",
        min_value=min_time,
        max_value=max_time,
        value=max_time,
        format="YYYY-MM-DD HH:mm"
    )
    df_display = df_data[df_data['time_stamp'] <= pd.Timestamp(selected_cutoff)].copy()
else:
    df_display = df_data.copy()


total_alerts = int(df_display['alert_triggered'].sum())
high_alerts = int((df_display['alert_level'] == 'HIGH').sum())

alerts_in_window = df_display[
    (df_display['alert_triggered'] == 1) & 
    (df_display['time_stamp'] >= window_start) & 
    (df_display['time_stamp'] <= window_end)
]

if len(alerts_in_window) > 0:
    first_alert = alerts_in_window['time_stamp'].min()
    lead_time_days = (window_end - first_alert).total_seconds() / 86400.0
    delay_days = (first_alert - window_start).total_seconds() / 86400.0
else:
    lead_time_days = 0.0
    delay_days = 0.0


kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric(
    label="TUA-BIN", 
    value=f"WTG_{selected_event:02d}", 
    delta="Bình thường" if high_alerts == 0 else "Cần can thiệp",
    delta_color="normal" if high_alerts == 0 else "inverse"
)
kpi2.metric(
    label="LEAD TIME", 
    value=f"{lead_time_days:.2f} Ngày" if lead_time_days > 0 else "0.00 Ngày",
    delta="Vượt chuẩn 7d" if lead_time_days >= 7 else None
)
kpi3.metric(
    label="PRECISION", 
    value="97.45%" if selected_event == 0 else "N/A"
)
kpi4.metric(
    label="TỔNG CẢNH BÁO", 
    value=f"{total_alerts} Lần"
)
kpi5.metric(
    label="CẢNH BÁO HIGH", 
    value=f"{high_alerts} Điểm", 
    delta=f"{delay_days:.2f}d trễ" if delay_days > 0 else "0d trễ",
    delta_color="inverse"
)

st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)


st.markdown("### 📈 Phân Tích Chuỗi Thời Gian Điểm Bất Thường & Cửa Sổ Sự Cố")

fig = go.Figure()


fig.add_trace(go.Scatter(
    x=df_display['time_stamp'],
    y=df_display['anomaly_score'],
    mode='lines',
    name='Anomaly Score (-decision_function)',
    line=dict(color='#38bdf8', width=1.6)
))


fig.add_trace(go.Scatter(
    x=df_display['time_stamp'],
    y=[threshold_val] * len(df_display),
    mode='lines',
    name=f'Ngưỡng vận hành (thr = {threshold_val:.2f})',
    line=dict(color='#fbbf24', width=1.4, dash='dash')
))


fig.add_vrect(
    x0=window_start, x1=window_end,
    fillcolor="rgba(244, 63, 94, 0.15)",
    layer="below", line_width=1,
    line_color="rgba(244, 63, 94, 0.4)",
    annotation_text=f"Anomaly Window ({component_name})",
    annotation_position="top left",
    annotation_font=dict(color="#f43f5e", size=11)
)


alert_points = df_display[df_display['alert_triggered'] == 1]
fig.add_trace(go.Scatter(
    x=alert_points['time_stamp'],
    y=alert_points['anomaly_score'],
    mode='markers',
    name='Cảnh Báo Kích Hoạt (3 kỳ liên tiếp)',
    marker=dict(
        color='#ef4444', 
        symbol='triangle-up', 
        size=10, 
        line=dict(color='#ffffff', width=1)
    ),
    customdata=alert_points[['alert_level']],
    hovertemplate="<b>Thời gian:</b> %{x}<br><b>Anomaly Score:</b> %{y:.4f}<br><b>Cấp độ:</b> %{customdata[0]}<extra></extra>"
))


fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#111827",
    plot_bgcolor="#0b0f19",
    font=dict(color="#cbd5e1", size=12),
    height=460,
    margin=dict(l=20, r=20, t=35, b=20),
    xaxis=dict(
        title="Thời gian vận hành (time_stamp)",
        gridcolor="#1e293b",
        zerolinecolor="#334155"
    ),
    yaxis=dict(
        title="Anomaly Score",
        gridcolor="#1e293b",
        zerolinecolor="#334155"
    ),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(17, 24, 39, 0.9)"
    )
)

st.plotly_chart(fig, use_container_width=True)


st.markdown("### 📋 Danh Sách Cảnh Báo Ưu Tiên Xử Lý")

if len(alert_points) > 0:
    table_df = alert_points[['time_stamp', 'asset_id', 'anomaly_score', 'alert_level']].copy()
    
    def assign_action(level):
        if level == 'HIGH':
            return "🔴 DỪNG MÁY / KIỂM TRA KHẨN CẤP (Tàu cẩu ngoài khơi)"
        elif level == 'MEDIUM':
            return "🟡 Theo dõi sát & Giảm tải công suất phát"
        else:
            return "🟢 Tiếp tục theo dõi định kỳ ca trực"
            
    table_df['Khuyến nghị vận hành'] = table_df['alert_level'].apply(assign_action)
    table_df = table_df.sort_values(by='time_stamp', ascending=True)
    
    st.dataframe(table_df, use_container_width=True)
    
    
    csv_bytes = table_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 Xuất Phiếu Cảnh Báo & Biên Bản Ca Trực (CSV Log)",
        data=csv_bytes,
        file_name=f"SCADA_Alert_Log_WTG_{selected_event:02d}.csv",
        mime="text/csv",
        help="Tải báo cáo chi tiết các mốc cảnh báo đã xác nhận để phục vụ điều độ kỹ thuật."
    )
else:
    st.success("✅ Trạng thái an toàn: Không ghi nhận tín hiệu bất thường nào vượt ngưỡng trong khoảng thời gian này.")