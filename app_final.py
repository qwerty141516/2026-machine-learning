import streamlit as st
import pandas as pd
import numpy as np
import random
import folium
import time

from streamlit_folium import st_folium
import xgboost as xgb

# =========================
# 🌐 PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Rent Intelligence",
    page_icon="🏠",
    layout="wide"
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
}

.big-number {
    font-size: 28px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏠 AI Rent Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>서울 원룸 가격 AI 분석 시스템</div>", unsafe_allow_html=True)

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

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

model.fit(rooms[features], rooms["rent"])

# =========================
# 🏫 SELECT (UI FIX 핵심)
# =========================
colA, colB = st.columns([1, 3])

with colA:
    selected = st.selectbox("🏫 대학 선택", list(UNIV.keys()))

with colB:
    st.markdown("")

filtered = rooms[rooms["univ"] == selected]
lat, lon = UNIV[selected]

# =========================
# 🗺 MAP (선택 아래 정렬 유지)
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

map_data = st_folium(m, height=650, width=None)

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

    input_data = pd.DataFrame([{
        "area": clicked["area"],
        "dist": clicked["dist"],
        "age": clicked["age"],
        "floor": clicked["floor"]
    }])

    pred = model.predict(input_data)[0]
    diff = ((pred - clicked["rent"]) / clicked["rent"]) * 100

    # =========================
    # 📊 DASHBOARD
    # =========================
    col1, col2, col3 = st.columns(3)

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
    # 🧠 AI 설명 (핵심 추가)
    # =========================
    st.markdown("## 🧠 AI 가격 설명")

    reasons = []

    if clicked["dist"] < 200:
        reasons.append("📍 학교와 매우 가까워 수요가 높습니다")
    elif clicked["dist"] > 600:
        reasons.append("📍 학교와 거리가 있어 가격이 낮아지는 요인입니다")

    if clicked["area"] > 25:
        reasons.append("🏠 면적이 넓어 가격이 상승합니다")
    else:
        reasons.append("🏠 작은 원룸 크기로 인해 가격이 낮습니다")

    if clicked["age"] > 15:
        reasons.append("🏚️ 건물이 오래되어 감가 요인이 있습니다")
    else:
        reasons.append("🏢 비교적 신축 건물입니다")

    if clicked["floor"] >= 5:
        reasons.append("🏙️ 중·고층으로 선호도가 반영됩니다")

    st.markdown(f"""
    ### 🤖 AI 분석 결과
    이 매물의 예상 월세는 **{pred:.1f}만원**이며,  
    현재 가격({clicked['rent']}만원)과 비교했을 때 **{diff:.1f}% 차이**가 있습니다.

    **주요 요인:**
    """)

    for r in reasons:
        st.write("- " + r)

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
