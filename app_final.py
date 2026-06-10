import streamlit as st
import pandas as pd
import numpy as np
import random
import folium
import time
import matplotlib.pyplot as plt

from streamlit_folium import st_folium
import xgboost as xgb
import shap

# =========================
# 🌐 PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Rent Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 🎨 UI STYLE
# =========================
st.markdown("""
<style>
.block-container { padding: 2rem; }

.main-title {
    font-size: 38px;
    font-weight: 800;
    text-align: center;
    color: white;
}

.sub-title {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 25px;
}

.card {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 18px;
    border-radius: 16px;
    color: white;
    margin-bottom: 10px;
}

.big-number {
    font-size: 28px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏠 AI Rent Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>SHAP 기반 서울 원룸 가격 분석 시스템</div>", unsafe_allow_html=True)

# =========================
# 🏫 DATA
# =========================
UNIV = {
    "연세대": (37.5658, 126.9386),
    "서울대": (37.4599, 126.9519),
    "고려대": (37.5894, 127.0324)
}

@st.cache_data
def make_rooms():
    rooms = []
    for u, (lat, lon) in UNIV.items():
        for i in range(30):
            rooms.append({
                "name": f"{u} 원룸 {i+1}",
                "univ": u,
                "lat": lat + random.uniform(-0.004, 0.004),
                "lon": lon + random.uniform(-0.004, 0.004),
                "area": random.randint(10, 40),
                "dist": random.randint(50, 900),
                "age": random.randint(1, 25),
                "floor": random.randint(1, 10),
                "rent": random.randint(35, 95)
            })
    return pd.DataFrame(rooms)

rooms = make_rooms()

# =========================
# 🧠 MODEL
# =========================
features = ["area", "dist", "age", "floor"]

X = rooms[features]
y = rooms["rent"]

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

model.fit(X, y)

explainer = shap.Explainer(model)

# =========================
# 🏫 SELECT
# =========================
selected = st.selectbox("🏫 대학 선택", list(UNIV.keys()))
filtered = rooms[rooms["univ"] == selected]

lat, lon = UNIV[selected]

# =========================
# 🗺 MAP
# =========================
m = folium.Map(location=[lat, lon], zoom_start=15, tiles="cartodbpositron")

for _, r in filtered.iterrows():
    folium.CircleMarker(
        location=[r["lat"], r["lon"]],
        radius=5,
        color="#3b82f6",
        fill=True,
        fill_opacity=0.8,
        tooltip=f"{r['name']} | {r['rent']}만원"
    ).add_to(m)

map_data = st_folium(m, height=650, width=1100)

# =========================
# 📌 CLICK DETECTION
# =========================
clicked = None

if map_data and map_data.get("last_object_clicked"):
    c = map_data["last_object_clicked"]

    temp = filtered.copy()
    temp["d"] = (temp["lat"] - c["lat"])**2 + (temp["lon"] - c["lng"])**2
    clicked = temp.sort_values("d").iloc[0]

# =========================
# 🚀 AI ANALYSIS
# =========================
st.markdown("---")

if clicked is not None:

    progress = st.progress(0)
    status = st.empty()

    status.write("🧠 AI 모델 분석 시작...")
    time.sleep(0.2)
    progress.progress(20)

    input_data = pd.DataFrame([{
        "area": clicked["area"],
        "dist": clicked["dist"],
        "age": clicked["age"],
        "floor": clicked["floor"]
    }])

    status.write("📊 가격 예측 계산 중...")
    pred = model.predict(input_data)[0]
    time.sleep(0.2)
    progress.progress(50)

    status.write("🔍 SHAP 설명 생성 중...")
    shap_values = explainer(input_data)
    time.sleep(0.3)
    progress.progress(80)

    status.write("📈 결과 정리 중...")
    time.sleep(0.2)
    progress.progress(100)

    status.write("완료!")
    time.sleep(0.3)

    progress.empty()
    status.empty()

    # =========================
    # 📊 DASHBOARD
    # =========================
    col1, col2, col3 = st.columns(3)

    diff = ((pred - clicked["rent"]) / clicked["rent"]) * 100

    with col1:
        st.markdown(f"""
        <div class="card">
            📍 현재 월세<br>
            <div class="big-number">{clicked['rent']}만원</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            🤖 AI 예측<br>
            <div class="big-number">{pred:.1f}만원</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            📊 차이율<br>
            <div class="big-number">{diff:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # 🧠 SHAP TEXT
    # =========================
    st.markdown("## 🧠 AI 가격 설명 (SHAP)")

    shap_array = shap_values.values[0]
    impact = list(zip(features, shap_array))
    impact.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, val in impact:
        icon = "📈" if val > 0 else "📉"
        st.write(f"{icon} **{name}** → {val:.2f}")

    # =========================
    # 📊 SHAP PLOT (FIXED)
    # =========================
    st.markdown("### 📊 SHAP 시각화")

    plt.clf()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())

    # =========================
    # 📊 MARKET
    # =========================
    avg = filtered["rent"].mean()

    st.markdown("## 📊 시장 비교")

    st.write(f"- 평균 월세: **{avg:.1f}만원**")

    if clicked["rent"] > avg:
        st.error("📈 시장 대비 고가 매물")
    else:
        st.success("📉 시장 대비 저가 매물")

else:
    st.info("지도에서 매물을 클릭하면 AI 분석이 시작됩니다.")
