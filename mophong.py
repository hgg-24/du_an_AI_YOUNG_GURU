import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from PIL import Image
from scipy.optimize import minimize_scalar

# [CẤU HÌNH & CSS]
st.set_page_config(page_title="Bài dự thi của Studyholics", layout="wide")

st.markdown("""
<style>
    .stApp, .stApp > header { background-color: #0A1128 !important; }
    
    p, label, h1, h2, h3, h4, li { color: #F8FAFC !important; font-family: sans-serif !important; }
    span.material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; color: #94A3B8 !important; }
    
    h1 { color: #00FFFF !important; text-shadow: 0 0 12px rgba(0, 255, 255, 0.6); text-align: center; text-transform: uppercase; margin-bottom: 30px;}
    [data-testid="stSidebar"] { background-color: #121833 !important; border-right: 1px solid #00FFFF !important; padding-top: 20px;}
    
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="textarea"] > div { background-color: #FFFFFF !important; border: 1px solid #94A3B8 !important; border-radius: 5px !important; }
    div[data-baseweb="input"] input, div[data-baseweb="select"] div, textarea { color: #000000 !important; font-weight: bold !important; }
    div[data-testid="stFileUploadDropzone"] * { color: #1E293B !important; }
    
    ul[role="listbox"], ul[role="listbox"] li, div[data-baseweb="popover"] * { background-color: #FFFFFF !important; color: #000000 !important; }
    
    div[data-testid="stTickBar"] { display: none !important; }
    div[data-baseweb="slider"] div[data-testid="stSliderTrack"] > div { background-color: #FFB6C1 !important; }
    div[data-baseweb="slider"] div[role="slider"] { background-color: #FFB6C1 !important; border: 2px solid #FF69B4 !important; }
    
    .stTabs [data-baseweb="tab"] { color: #94A3B8 !important; }
    .stTabs [aria-selected="true"] { color: #00FFFF !important; border-bottom: 2px solid #00FFFF !important; background-color: transparent !important; }

    div[data-baseweb="select"] span { color: #000000 !important; }
    div[data-baseweb="input"] svg, div[data-baseweb="select"] svg { fill: #000000 !important; color: #000000 !important; }
    p.dual-label { margin-bottom: 2px !important; margin-top: 10px !important; font-weight: bold !important; color: #00FFFF !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>MÔ PHỎNG CHUYỂN ĐỘNG NÉM</h1>", unsafe_allow_html=True)

# [HÀM NHẬP LIỆU]
def dual_input(label, key, min_val, max_val, default_val, step=0.1):
    if key not in st.session_state: st.session_state[key] = float(default_val)
    def update_num(): st.session_state[key] = st.session_state[f"n_{key}"]
    def update_sli(): st.session_state[key] = st.session_state[f"s_{key}"]
    st.markdown(f"<p class='dual-label'>{label}</p>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2.5])
    with c1: st.number_input(label, min_value=float(min_val), max_value=float(max_val), value=float(st.session_state[key]), step=float(step), key=f"n_{key}", on_change=update_num, label_visibility="collapsed")
    with c2: st.slider(label, min_value=float(min_val), max_value=float(max_val), value=float(st.session_state[key]), step=float(step), key=f"s_{key}", on_change=update_sli, label_visibility="collapsed")
    return st.session_state[key]

# [ENGINE VẬT LÝ]
@st.cache_data
def calc_trajectory(v0, h0, alpha_deg, g, has_drag, m=1.0):
    k = 0.05 if has_drag else 0.0
    dt = 0.005 
    alpha_rad = np.radians(alpha_deg)
    vx, vy = v0 * np.cos(alpha_rad), v0 * np.sin(alpha_rad)
    x, y, t = 0.0, h0, 0.0
    
    data = {"t": [t], "x": [x], "y": [y], "v": [v0], "Wd": [0.5 * m * v0**2], "Wt": [m * g * y]}
    
    while y >= 0:
        v = np.sqrt(vx**2 + vy**2)
        ax = -(k/m)*v*vx if has_drag else 0
        ay = -g - (k/m)*v*vy if has_drag else -g
        
        y_prev, vy_prev = y, vy
        
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
        
        if y < 0:
            discriminant = vy_prev**2 - 4 * (0.5 * ay) * y_prev
            if discriminant >= 0:
                dt_exact = (-vy_prev - np.sqrt(discriminant)) / ay
            else:
                dt_exact = dt * (abs(y_prev) / (abs(y_prev) + abs(y))) 
            
            x = data["x"][-1] + data["v"][-1] * (vx / np.sqrt(vx**2 + vy**2)) * dt_exact
            t = data["t"][-1] + dt_exact
            y = 0.0
            vx, vy = 0, 0
            v_current = 0
        else:
            v_current = np.sqrt(vx**2 + vy**2)
            
        data["t"].append(t)
        data["x"].append(x)
        data["y"].append(y)
        data["v"].append(v_current)
        data["Wd"].append(0.5 * m * v_current**2)
        data["Wt"].append(m * g * y)
        
        if y == 0: break

    return pd.DataFrame({
        "Thời gian (s)": data["t"],
        "X (m)": data["x"],
        "Y (m)": data["y"],
        "Vận tốc (m/s)": data["v"],
        "Động năng (J)": data["Wd"],
        "Thế năng (J)": data["Wt"]
    })

# [TỐI ƯU HÓA TOÁN HỌC - SCIPY]
def _objective_func(alpha_deg, v0, h0, g, has_drag, m=1.0):
    k = 0.05 if has_drag else 0.0
    dt = 0.02
    alpha_rad = np.radians(alpha_deg)
    vx, vy = v0 * np.cos(alpha_rad), v0 * np.sin(alpha_rad)
    x, y = 0.0, h0
    while y >= 0:
        v = np.sqrt(vx**2 + vy**2)
        vx += (-(k/m)*v*vx if has_drag else 0) * dt
        vy += (-g - (k/m)*v*vy if has_drag else -g) * dt
        x += vx * dt
        y += vy * dt
    return -x 

@st.cache_data
def optimize_angle(v0, h0, g, has_drag):
    res = minimize_scalar(
        _objective_func, bounds=(0, 90), 
        args=(v0, h0, g, has_drag), method='bounded'
    )
    return res.x, -res.fun

# [SIDEBAR CÀI ĐẶT]
with st.sidebar:
    st.markdown("<h2 style='color: #00FFFF;'>⚙️ CÀI ĐẶT THÔNG SỐ</h2>", unsafe_allow_html=True)
    g_options = {"Trái Đất (9.81 m/s²)": 9.81, "Mặt Trăng (1.62 m/s²)": 1.62, "Sao Hỏa (3.71 m/s²)": 3.71, "Sao Mộc (24.79 m/s²)": 24.79}
    g_choice = st.selectbox("Môi trường", list(g_options.keys()))
    g = g_options[g_choice]
    
    has_drag = st.checkbox("Bật Lực cản không khí (k = 0.05)")
    
    v0_mode = st.radio("Cách tính Vận tốc đầu (v0)", ["Nhập trực tiếp", "Từ máng trượt"])
    if v0_mode == "Nhập trực tiếp":
        v0 = dual_input("Vận tốc đầu v0 (m/s)", "v0_val", 0, 100, 15)
    else:
        h_mang = dual_input("Độ cao máng trượt (m)", "h_mang", 0, 50, 5)
        v0 = np.sqrt(2 * g * h_mang)
        st.info(f"Vận tốc v0 = {v0:.2f} m/s")
        
    h0 = dual_input("Độ cao ban đầu h0 (m)", "h0_val", 0, 100, 10)
    
    throw_mode = st.radio("Chế độ ném", ["Ném ngang", "Ném xiên"])
    if throw_mode == "Ném ngang":
        alpha = 0.0
        st.info("Góc ném bị khóa ở 0°")
    else:
        auto_opt = st.checkbox("Bật Tự động tìm Góc ném tối ưu")
        if auto_opt:
            with st.spinner("Đang xử lý thuật toán tối ưu (Scipy)..."):
                alpha, max_dist = optimize_angle(v0, h0, g, has_drag)
            st.success(f"Góc hội tụ: {alpha:.4f}° (Tầm xa tối đa: {max_dist:.4f}m)")
        else:
            alpha = dual_input("Góc ném α (độ)", "alpha_val", -90, 90, 45)

    st.markdown("---")
    st.markdown("<h3 style='color: #FF007F;'>🎮 MỤC TIÊU TRÒ CHƠI</h3>", unsafe_allow_html=True)
    target_x = dual_input("Tọa độ X mục tiêu (m)", "tx", 1, 150, 20)
    target_y = dual_input("Tọa độ Y mục tiêu (m)", "ty", 0, 50, 0)

# [TẠO DỮ LIỆU & ĐỒ THỊ GLOBAL]
df = calc_trajectory(v0, h0, alpha, g, has_drag)

fig = go.Figure()

if has_drag:
    df_ideal = calc_trajectory(v0, h0, alpha, g, has_drag=False)
    fig.add_trace(go.Scatter(
        x=df_ideal["X (m)"], y=df_ideal["Y (m)"], mode="lines",
        line=dict(color="#FDE047", width=2, dash="dash"), 
        name="Lý tưởng (k=0)",
        hovertemplate="[Lý tưởng]<br>t: %{customdata[0]:.3f} s<br>X: %{x:.3f} m<br>Y: %{y:.3f} m",
        customdata=df_ideal[["Thời gian (s)"]].values
    ))

trace_name = "Thực tế (k=0.05)" if has_drag else "Quỹ đạo"
fig.add_trace(go.Scatter(
    x=df["X (m)"], y=df["Y (m)"], mode="lines",
    line=dict(color="#00FFFF", width=3), name=trace_name,
    hovertemplate="t: %{customdata[0]:.3f} s<br>X: %{x:.3f} m<br>Y: %{y:.3f} m<br>v: %{customdata[1]:.3f} m/s<br>Wđ: %{customdata[2]:.2f} J<br>Wt: %{customdata[3]:.2f} J",
    customdata=df[["Thời gian (s)", "Vận tốc (m/s)", "Động năng (J)", "Thế năng (J)"]].values
))

fig.add_trace(go.Scatter(
    x=[target_x], y=[target_y], mode="markers",
    marker=dict(color="#FF007F", symbol="star", size=15, line=dict(color="white", width=1)),
    name="Mục tiêu"
))

fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title="Tọa độ X (m)", gridcolor="#1E293B", zerolinecolor="#1E293B"),
    yaxis=dict(title="Tọa độ Y (m)", gridcolor="#1E293B", zerolinecolor="#1E293B"),
    font=dict(color="#F8FAFC"),
    legend=dict(font=dict(color="#F8FAFC")), 
    margin=dict(l=20, r=20, t=30, b=20)
)

# [RENDER TABS]
tab1, tab2 = st.tabs(["🚀 MÔ PHỎNG & TRÒ CHƠI", "🧠 TRỢ GIẢNG AI"])

with tab1:
    L = df["X (m)"].iloc[-1]
    t_flight = df["Thời gian (s)"].iloc[-1]
    H_max = df["Y (m)"].max()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Tầm xa L (m)", f"{L:.4f}")
    c2.metric("Thời gian bay t (s)", f"{t_flight:.4f}")
    c3.metric("Độ cao cực đại H (m)", f"{H_max:.4f}")
    
    # ---- BẢN VÁ LOGIC PHÁO HOA ----
    distances = np.sqrt((df["X (m)"] - target_x)**2 + (df["Y (m)"] - target_y)**2)
    if distances.min() <= 1.0:
        hit_id = f"{target_x}_{target_y}_{v0}_{alpha}_{has_drag}" 
        if st.session_state.get("last_hit") != hit_id:
            st.balloons()
            st.session_state["last_hit"] = hit_id
        st.success("🎉 Chúc mừng! Quỹ đạo đã trúng mục tiêu!")
    else:
        # Nếu trượt mục tiêu, xóa trí nhớ cũ đi để lần sau bắn trúng lại vẫn có pháo hoa
        st.session_state["last_hit"] = None
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 📊 Bảng Số Liệu Chi Tiết (Theo quỹ đạo thực tế)")
    st.dataframe(df.round(4), use_container_width=True, height=200)

with tab2:
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception as e:
        st.error("Chưa cấu hình API Key trong Streamlit Secrets!")
    
    st.markdown("<h3 style='color:#00FFFF;'>🤖 Gia sư AI - Giải đáp Vật lý & Code</h3>", unsafe_allow_html=True)
    c_chat, c_graph = st.columns([1.2, 1])
    
    with c_chat:
        uploaded_file = st.file_uploader("Tải ảnh đề bài/Code lỗi", type=["jpg", "png", "jpeg"])
        if uploaded_file: st.image(Image.open(uploaded_file), caption="Đề bài", use_container_width=True)
            
        q = st.text_area("Hỏi Gia sư (Không giải bài hộ, chỉ gợi ý):", height=150)
        
        if st.button("Gửi câu hỏi", type="primary", use_container_width=True):
            if uploaded_file and q:
                try:
                    with st.spinner("Gia sư đang phân tích..."):
                        model = genai.GenerativeModel("gemini-1.5-flash", system_instruction="Bạn là gia sư Vật lý 10 nghiêm khắc. Chỉ gợi ý phương pháp, giải thích hiện tượng. KHÔNG giải ra đáp án cuối cùng.")
                        res = model.generate_content([q, Image.open(uploaded_file)])
                        st.success("Phản hồi từ Gia Sư:")
                        st.write(res.text)
                except Exception as e:
                    st.error(f"⚠️ Lỗi kết nối API: {e}")
            else:
                st.warning("Vui lòng tải ảnh đề bài và nhập câu hỏi!")
                
    with c_graph:
        st.markdown("**📈 Đối chiếu với Đồ thị Mô phỏng**")
        st.caption("Theo dõi đồ thị quỹ đạo hiện tại để đối chiếu với gợi ý của Gia sư")
        st.plotly_chart(fig, use_container_width=True, key="graph_tab2")
