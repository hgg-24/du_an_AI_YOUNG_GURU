import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from PIL import Image
from scipy.optimize import minimize_scalar

# [CẤU HÌNH TRANG]
st.set_page_config(page_title="Bài dự thi của Studyholics", layout="wide")

# [CSS - BẢN FIX CUỐI CÙNG CHO TOÀN BỘ LỖI UI]
st.markdown("""
<style>
    /* --- 1. CẤU HÌNH CHUNG NỀN TỐI --- */
    .stApp, .stApp > header { background-color: #0A1128 !important; }
    
    /* VÁ LỖI ẢNH 1: Bỏ thẻ 'span' và 'div' ra khỏi lệnh ép font toàn cục để không hỏng Icon Streamlit */
    p, label, li, h1, h2, h3, h4, h5, h6 { 
        color: #FFFFFF; 
        font-family: 'Verdana', sans-serif; 
    }
    
    /* Cứu lại bộ font Icon của Streamlit */
    .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; color: #FFFFFF !important; }
    
    /* Tiêu đề chính Neon */
    h1 { color: #00FFFF !important; text-shadow: 0 0 15px #00FFFF; text-transform: uppercase; text-align: center; font-weight: 900 !important; }
    div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] { display: none !important; }

    /* --- 2. XỬ LÝ DROPDOWN & INPUT SỐ (NỀN TRẮNG - CHỮ ĐEN) --- */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important; 
        border: 2px solid #00FFFF !important;
    }
    div[data-baseweb="select"] *, div[data-baseweb="input"] input {
        color: #000000 !important; 
        fill: #000000 !important;
        font-weight: bold !important;
    }
    div[data-baseweb="popover"], ul[data-baseweb="menu"], ul[data-baseweb="menu"] li {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
    ul[data-baseweb="menu"] li:hover { background-color: #E2E8F0 !important; }

    /* --- 3. VÁ LỖI ẢNH 3, 4: KHUNG UPLOAD & TEXT AREA (NỀN TRẮNG - CHỮ ĐEN) --- */
    [data-testid='stFileUploadDropzone'] {
        background-color: #FFFFFF !important; 
        border: 2px dashed #00FFFF !important; 
        padding: 20px;
    }
    [data-testid='stFileUploadDropzone'] * { color: #000000 !important; font-weight: bold !important; }
    [data-testid='stFileUploadDropzone'] svg { fill: #000000 !important; width: 3rem !important; height: 3rem !important; }
    [data-testid='stFileUploadDropzone'] button {
        background-color: #00FFFF !important; color: #000000 !important; border: none !important; font-weight: 900 !important;
    }
    /* Text Area của AI */
    div[data-baseweb="textarea"] > div, div[data-baseweb="textarea"] textarea {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: bold !important;
    }

    /* --- 4. VÁ LỖI ẢNH 5: KHUNG HIỂN THỊ CÂU TRẢ LỜI CỦA AI --- */
    /* Tạo class riêng để giữ màu nền trắng chữ đen nhưng không làm hỏng LaTeX */
    .ai-response-box {
        background-color: #FFFFFF; 
        border: 2px solid #00FFFF; 
        border-radius: 8px; 
        padding: 20px; 
        margin-top: 15px;
    }
    .ai-response-box p, .ai-response-box li, .ai-response-box div, .ai-response-box span {
        color: #000000 !important;
    }
    .ai-response-box .katex * { color: #D90429 !important; font-weight: bold; } /* Đỏ đậm cho công thức Lý */

    /* --- 5. CÁC THÀNH PHẦN KHÁC --- */
    [data-testid="stSidebar"] { background-color: #111827 !important; border-right: 2px solid #00FFFF; }
    div[data-baseweb="slider"] div[role="slider"] { background-color: #FF007F !important; border: 2px solid white; }
    .stTabs [data-baseweb="tab"] { color: #CBD5E1 !important; font-weight: bold; }
    .stTabs [aria-selected="true"] { color: #00FFFF !important; border-bottom-color: #00FFFF !important; }

    /* Hộp Expander */
    div[data-testid="stExpander"] details > div { background-color: #FFFFFF !important; border: 2px solid #00FFFF !important; border-top: none; color: #000000 !important; }
    div[data-testid="stExpander"] details > div * { color: #000000 !important; }
    div[data-testid="stExpander"] details > div h4 { color: #00008B !important; font-weight: bold !important; }
    div[data-testid="stExpander"] details > summary { background-color: #F8FAFC !important; border: 2px solid #00FFFF !important; border-radius: 5px 5px 0 0; color: #0A1128 !important; }
    div[data-testid="stExpander"] details > summary svg { display: none !important; }
    div[data-testid="stExpander"] details > summary::before { content: "▶" !important; color: #0A1128 !important; font-size: 1.2rem !important; margin-right: 10px !important; display: inline-block; transition: transform 0.3s ease; }
    div[data-testid="stExpander"] details[open] > summary::before { transform: rotate(90deg) !important; }
    div[data-testid="stExpander"] details > summary p { color: #0A1128 !important; font-weight: 900 !important; font-size: 1.2rem !important; display: inline; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚀 MÔ PHỎNG CHUYỂN ĐỘNG NÉM</h1>", unsafe_allow_html=True)

# --- [PHẦN 1] HƯỚNG DẪN SỬ DỤNG ---
with st.expander("📖 HƯỚNG DẪN SỬ DỤNG (QUY TRÌNH CHUẨN)", expanded=False):
    st.markdown("#### 🔻 Sơ đồ luồng hoạt động:")
    st.graphviz_chart('''
    digraph {
        rankdir=LR;
        bgcolor="white"; 
        
        node [shape=box, style="filled,rounded", fillcolor="#E0F2FE", fontname="Verdana", fontsize=11, fontcolor="black", penwidth=1, color="#0284C7"];
        edge [color="#334155", penwidth=2, arrowsize=1.0]; 
        
        Start [label="BẮT ĐẦU", shape=circle, fillcolor="#F472B6", fontcolor="white", width=1.0, style=filled];
        SetEnv [label="1. Chọn Môi trường\n(Trái Đất/Sao Hỏa...)"];
        SetParams [label="2. Nhập thông số\n(v0, góc, độ cao)"];
        View [label="Xem Đồ thị & Số liệu", fillcolor="#FEF08A"];
        Target [label="3. Đặt Mục tiêu (X, Y)"];
        Check [label="Trúng đích?\n(Sai số < 1m)", shape=diamond, fillcolor="#FDE047"];
        Win [label="PHÁO HOA! 🎉", shape=star, fillcolor="#EF4444", fontcolor="white", fontsize=14];
        Adjust [label="Chỉnh lại v0, góc", fillcolor="#E5E7EB"];
        
        AI [label="4. Bí bài?\nHỏi Gia sư AI (Tab 2)", shape=note, style=filled, fillcolor="#1E293B", fontcolor="#00FFFF"];

        Start -> SetEnv -> SetParams -> View;
        View -> Target -> Check;
        Check -> Win [label="CÓ", fontcolor="#15803d", fontsize=10];
        Check -> Adjust [label="KHÔNG", fontcolor="#b91c1c", fontsize=10];
        Adjust -> SetParams;
        Adjust -> AI [style=dashed, color="#0EA5E9"];
    }
    ''')

# --- [PHẦN 2] GÓC HỌC TẬP ---
with st.expander("📘 GÓC HỌC TẬP: CÔNG THỨC & THUẬT TOÁN", expanded=False):
    tab_lythuyet, tab_thuatToan = st.tabs(["📚 VẬT LÝ 10", "💻 KHOA HỌC MÁY TÍNH"])
    
    with tab_lythuyet:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 1. Môi trường Lý tưởng (Chân không)")
            st.info("Bỏ qua lực cản. Vật chỉ chịu tác dụng của Trọng lực (P).")
            st.latex(r"\begin{cases} x = v_0 \cos(\alpha) \cdot t \\ y = h_0 + v_0 \sin(\alpha) \cdot t - \frac{1}{2}gt^2 \end{cases}")
            
        with c2:
            st.markdown("#### 2. Môi trường Thực tế (Có gió)")
            st.warning("Có lực cản không khí tỉ lệ với bình phương vận tốc. Quỹ đạo sẽ bị méo.")
            st.latex(r"\vec{F}_c = -k \cdot v \cdot \vec{v}")

        st.markdown("---")
        st.markdown("#### 3. Định luật Bảo toàn Năng lượng")
        st.latex(r"W = W_đ + W_t = \frac{1}{2}mv^2 + mgy")
    
    with tab_thuatToan:
        st.markdown("#### 1. Tại sao máy tính vẽ được đường cong?")
        st.success("Máy tính dùng phương pháp số **Euler** để tính toán từng bước nhảy siêu nhỏ (dt = 0.005s).")
        st.code("""
# Thuật toán Euler (Python Code)
vx_moi = vx_cu + ax * dt   # Tính vận tốc mới
vy_moi = vy_cu + ay * dt
x_moi  = x_cu  + vx_moi * dt # Tính tọa độ mới
y_moi  = y_cu  + vy_moi * dt
        """, language="python")
        
        st.markdown("#### 2. AI tìm góc bắn tối ưu thế nào?")
        st.info("Hệ thống sử dụng thuật toán tối ưu hóa **Golden-section Search** (thư viện `scipy`) để tìm góc bắn xa nhất.")
        st.latex(r"\alpha_{opt} = \arg \max (L)")

# [HÀM NHẬP LIỆU KÉP]
def dual_input(label, key, min_val, max_val, default_val, step=0.1):
    if key not in st.session_state: st.session_state[key] = float(default_val)
    def update_num(): st.session_state[key] = st.session_state[f"n_{key}"]
    def update_sli(): st.session_state[key] = st.session_state[f"s_{key}"]
    st.markdown(f"<p style='color:#FFFFFF; margin-bottom:2px; font-weight:bold;'>{label}</p>", unsafe_allow_html=True)
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
    target_x = dual_input("Tọa độ X mục tiêu (m)", "tx", 1, 150, 50) 
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
    line=dict(color="#00FFFF", width=3), 
    name=trace_name,
    hovertemplate="t: %{customdata[0]:.3f} s<br>X: %{x:.3f} m<br>Y: %{y:.3f} m<br>v: %{customdata[1]:.3f} m/s<br>Wđ: %{customdata[2]:.2f} J<br>Wt: %{customdata[3]:.2f} J",
    customdata=df[["Thời gian (s)", "Vận tốc (m/s)", "Động năng (J)", "Thế năng (J)"]].values
))

fig.add_trace(go.Scatter(
    x=[target_x], y=[target_y], mode="markers",
    marker=dict(color="#FF007F", symbol="star", size=15, line=dict(color="white", width=2)), 
    name="Mục tiêu"
))

# VÁ LỖI ẢNH 2: Ép màu trắng trực tiếp cho Legend
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color="#FFFFFF"), 
    xaxis=dict(title="Tọa độ X (m)", gridcolor="#475569", zerolinecolor="#FFFFFF"),
    yaxis=dict(title="Tọa độ Y (m)", gridcolor="#475569", zerolinecolor="#FFFFFF"),
    legend=dict(font=dict(color="#FFFFFF"), orientation="h", y=1.1), 
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
    
    distances = np.sqrt((df["X (m)"] - target_x)**2 + (df["Y (m)"] - target_y)**2)
    if distances.min() <= 1.0:
        hit_id = f"{target_x}_{target_y}_{v0}_{alpha}_{has_drag}" 
        if st.session_state.get("last_hit") != hit_id:
            st.balloons()
            st.session_state["last_hit"] = hit_id
        st.success("🎉 Chúc mừng! Quỹ đạo đã trúng mục tiêu!")
    else:
        st.session_state["last_hit"] = None
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 📊 Bảng Số Liệu Chi Tiết (Theo quỹ đạo thực tế)")
    st.dataframe(df.round(4), use_container_width=True, height=200)

with tab2:
    api_ready = False
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        api_ready = True
    except Exception as e:
        st.error("⚠️ Chưa cấu hình API Key trong Streamlit Secrets! Tính năng Trợ giảng AI tạm khóa.")
    
    st.markdown("<h3 style='color:#00FFFF;'>🤖 Gia sư AI - Giải đáp Vật lý & Code</h3>", unsafe_allow_html=True)
    c_chat, c_graph = st.columns([1.2, 1])
    
    with c_chat:
        uploaded_file = st.file_uploader("Tải ảnh đề bài/Code lỗi", type=["jpg", "png", "jpeg"])
        if uploaded_file: st.image(Image.open(uploaded_file), caption="Đề bài", use_container_width=True)
            
        q = st.text_area("Hỏi Gia sư (Không giải bài hộ, chỉ gợi ý):", height=150)
        
        if st.button("Gửi câu hỏi", type="primary", use_container_width=True, disabled=not api_ready):
            if uploaded_file and q:
                try:
                    with st.spinner("Gia sư đang phân tích..."):
                        model = genai.GenerativeModel("gemini-1.5-flash-latest", system_instruction="Bạn là gia sư Vật lý 10 nghiêm khắc. Chỉ gợi ý phương pháp, giải thích hiện tượng. KHÔNG giải ra đáp án cuối cùng.")
                        res = model.generate_content([q, Image.open(uploaded_file)])
                        st.success("Phản hồi từ Gia Sư:")
                        
                        # VÁ LỖI ẢNH 5: Dùng st.markdown kết hợp với CSS div để KHÔNG hỏng định dạng Markdown & LaTeX
                        st.markdown('<div class="ai-response-box">', unsafe_allow_html=True)
                        st.markdown(res.text) # Render nguyên bản
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"⚠️ Lỗi kết nối API: {e}")
            else:
                st.warning("Vui lòng tải ảnh đề bài và nhập câu hỏi!")
                
    with c_graph:
        st.markdown("**📈 Đối chiếu với Đồ thị Mô phỏng**")
        st.caption("Theo dõi đồ thị quỹ đạo hiện tại để đối chiếu với gợi ý của Gia sư")
        st.plotly_chart(fig, use_container_width=True, key="graph_tab2")

