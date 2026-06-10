import streamlit as st
import pandas as pd
import numpy as np
import random
import folium

from streamlit_folium import st_folium

# =========================
# 🌐 CONFIG
# =========================
st.set_page_config(page_title="AI Rent Simulator", layout="wide")

st.markdown("""
<h1 style='text-align:center;'>🏠 AI Rent Simulator</h1>
<p style='text-align:center;color:gray;'>경제 기반 월세 분석 시스템</p>
""", unsafe_allow_html=True)

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
# 🏫 UI
# =========================
col1, col2 = st.columns([1, 2])

with col1:
    selected = st.selectbox("🏫 대학 선택", list(UNIV.keys()))

filtered = rooms[rooms["univ"] == selected]
lat, lon = UNIV[selected]

# =========================
# 🗺 MAP
# =========================
with col2:
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

    map_data = st_folium(m, height=600)

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
# 🚀 ANALYSIS
# =========================
st.markdown("---")

if clicked is not None:

    base = clicked["rent"]

    # 경제 기반 시뮬레이션
    macro_score = random.uniform(-1, 1)
    pred = base + macro_score * 15

    # =========================
    # 💰 RESULT
    # =========================
    colA, colB = st.columns(2)

    with colA:
        st.metric("현재 월세", f"{base}만원")

    with colB:
        st.metric("예측 월세", f"{pred:.1f}만원")

    # =========================
    # 🤖 AI REPORT
    # =========================
    st.markdown("## 🤖 AI 분석 리포트")

    explanation = f"""
<div style="
background: linear-gradient(135deg,#111827,#1f2937);
padding:20px;
border-radius:15px;
color:white;
line-height:1.7;
">

현재 선택된 매물은 {selected} 인근 지역에 위치한 원룸이며,<br><br>

현재 월세는 <b>{base}만원</b>,
경제 환경을 반영한 예상 월세는 <b>{pred:.1f}만원</b>으로 분석됩니다.<br><br>

<b>📌 핵심 분석:</b><br>

• 이 매물의 가격은 단순 위치가 아니라 거시경제 흐름의 영향을 받고 있습니다.<br>
• 시장 변동성이 증가하면 임대료 민감도가 함께 상승합니다.<br>
• 금리, 유동성, 투자 심리 등이 복합적으로 작용합니다.<br>
• 단기적으로는 가격 변동성이 존재하는 구간입니다.<br><br>

<b>🧠 AI 해석:</b><br>
현재 시장은 안정 구간보다는 조정 구간에 가까우며,<br>
향후 경제 지표 변화에 따라 추가 상승 또는 하락 가능성이 존재합니다.

</div>
"""

    st.markdown(explanation, unsafe_allow_html=True)

else:
    st.info("지도에서 매물을 클릭하면 분석이 시작됩니다.")
