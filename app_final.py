import streamlit as st
import pandas as pd
import numpy as np
import random
import folium

from streamlit_folium import st_folium
import xgboost as xgb
import shap

# =========================
# 🎨 UI
# =========================
st.set_page_config(page_title="서울 주요 대학 근처 월세 예측 (AI+SHAP)", layout="wide")

st.markdown("""
<style>
.block-container {
    max-width: 1200px;
    margin: auto;
    padding-top: 2rem;
}

.title {
    text-align:center;
    font-size: 34px;
    font-weight: 800;
}

.sub {
    text-align:center;
    color: gray;
    margin-bottom: 20px;
}

.card {
    background:#111827;
    padding:15px;
    border-radius:14px;
    color:white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🏠 서울 주요 대학 근처 월세 예측</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>SHAP 기반 AI 가격 설명 시스템</div>", unsafe_allow_html=True)

# =========================
# 🏫 UNIVERSITY
# =========================
UNIV = {
    "연세대": (37.5658, 126.9386),
    "서울대": (37.4599, 126.9519),
    "고려대": (37.5894, 127.0324)
}

# =========================
# 🏠 DATA
# =========================
@st.cache_data
def make_rooms():
    rooms = []
    for u, (lat, lon) in UNIV.items():
        for i in range(25):
            rooms.append({
                "name": f"{u} 원룸 {i+1}",
                "univ": u,
                "lat": lat + random.uniform(-0.004, 0.004),
                "lon": lon + random.uniform(-0.004, 0.004),
                "area": random.randint(10, 35),
                "dist": random.randint(50, 800),
                "age": random.randint(1, 25),
                "floor": random.randint(1, 10),
                "rent": random.randint(35, 90)
            })
    return pd.DataFrame(rooms)

rooms = make_rooms()

# =========================
# 🧠 ML MODEL TRAINING
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

# =========================
# 📍 MAP
# =========================
lat, lon = UNIV[selected]
m = folium.Map(location=[lat, lon], zoom_start=15)

for _, r in filtered.iterrows():
    folium.CircleMarker(
        location=[r["lat"], r["lon"]],
        radius=5,
        color="blue",
        fill=True,
        fill_opacity=0.7,
        tooltip=f"{r['name']} | {r['rent']}만원"
    ).add_to(m)

map_data = st_folium(m, height=600, width=1100)

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
# 📊 AI ANALYSIS (SHAP)
# =========================
st.markdown("---")

if clicked is not None:

    input_data = pd.DataFrame([{
        "area": clicked["area"],
        "dist": clicked["dist"],
        "age": clicked["age"],
        "floor": clicked["floor"]
    }])

    pred_price = model.predict(input_data)[0]
    shap_values = explainer(input_data)

    st.markdown("## 🤖 AI 분석 리포트 (SHAP 기반)")

    st.write(f"""
📍 현재 월세: **{clicked['rent']}만원**  
📊 AI 예측 월세: **{pred_price:.1f}만원**
""")

    # =========================
    # 🔍 SHAP 설명
    # =========================
    st.markdown("### 🔍 가격 결정 요인 (AI 해석)")

    shap_array = shap_values.values[0]
    impact = list(zip(features, shap_array))
    impact.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, val in impact:
        if val > 0:
            st.write(f"📈 {name}: +{val:.2f}")
        else:
            st.write(f"📉 {name}: {val:.2f}")

    # =========================
    # 📊 비교
    # =========================
    avg_rent = filtered["rent"].mean()

    st.markdown("### ⚖️ 시장 비교")

    st.write(f"- 평균 월세: **{avg_rent:.1f}만원**")
    st.write("상태: " + ("고평가" if clicked["rent"] > avg_rent else "저평가"))

    # =========================
    # 💰 결과 카드
    # =========================
    st.markdown(f"""
    <div class="card">
        📊 AI 예측 월세: <b>{pred_price:.1f}만원</b>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 🧠 투자 점수
    # =========================
    score = 50
    if clicked["dist"] < 300: score += 20
    if clicked["age"] < 5: score += 15
    if abs(pred_price - clicked["rent"]) < 5: score += 10

    st.metric("🧠 투자 점수", f"{score}/100")

    # =========================
    # 📊 SHAP Waterfall
    # =========================
    st.markdown("### 📊 SHAP 시각화")

    fig = shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

else:
    st.info("지도에서 매물을 클릭하면 AI 분석이 시작됩니다.")
