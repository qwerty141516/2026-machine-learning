import streamlit as st
import pandas as pd
import numpy as np
import random
import folium

from streamlit_folium import st_folium

# =========================
# 🌐 CONFIG
# =========================
st.set_page_config(page_title="Macro Rent Simulator", layout="wide")

st.markdown("""
<h1 style='text-align:center;'>🏠 Macro Economic Rent Simulator</h1>
<p style='text-align:center;color:gray;'>환율 · 금리 · 물가 · 수요 기반 월세 변동 시뮬레이션</p>
""", unsafe_allow_html=True)

# =========================
# 🏫 UNIVERSITY DATA
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
                "rent": random.randint(40, 90)
            })
    return pd.DataFrame(rooms)

rooms = make_rooms()

# =========================
# 🌍 MACRO ECONOMY (가상 생성)
# =========================
macro = {
    "interest": round(random.uniform(2.0, 6.0), 2),   # 금리
    "exchange": round(random.uniform(1200, 1550), 0), # 환율
    "inflation": round(random.uniform(1.0, 5.0), 2),  # 물가
    "demand": round(random.uniform(0.7, 1.3), 2),     # 수요
    "supply": round(random.uniform(0.7, 1.3), 2)      # 공급
}

# =========================
# 🧠 PRICE MODEL (핵심)
# =========================
def predict_rent(base, m):
    return max(20,
        base
        + (m["interest"] - 3) * 6
        + (m["exchange"] - 1300) * 0.02
        + (m["inflation"] - 2) * 5
        + (m["demand"] - 1) * 15
        - (m["supply"] - 1) * 15
    )

# =========================
# 🏫 UI
# =========================
col1, col2 = st.columns([1, 2])

with col1:
    selected = st.selectbox("🏫 대학 선택", list(UNIV.keys()))

    st.markdown("## 🌍 가상 경제 상태")
    st.write(f"📈 금리: {macro['interest']}%")
    st.write(f"💱 환율: {macro['exchange']}")
    st.write(f"📊 물가: {macro['inflation']}%")
    st.write(f"🏙 수요: {macro['demand']}")
    st.write(f"🏗 공급: {macro['supply']}")

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
    pred = predict_rent(base, macro)

    # =========================
    # 💰 RESULT
    # =========================
    colA, colB = st.columns(2)

    with colA:
        st.metric("현재 월세", f"{base}만원")

    with colB:
        st.metric("예측 월세", f"{pred:.1f}만원")

    # =========================
    # 🤖 AI EXPLANATION (긴 GPT 스타일)
    # =========================
    st.markdown("## 🤖 AI 경제 분석 리포트")

    explanation = f"""
<div style="
background: linear-gradient(135deg,#111827,#1f2937);
padding:20px;
border-radius:15px;
color:white;
line-height:1.7;
">

현재 선택된 매물은 {selected} 인근 원룸이며,<br><br>

현재 월세는 <b>{base}만원</b>,
거시경제 조건을 반영한 예상 월세는 <b>{pred:.1f}만원</b>입니다.<br><br>

<b>🌍 경제 변수 분석:</b><br>

• 금리 {macro['interest']}%는 대출 비용 증가로 이어져 월세 상승 압력을 형성합니다.<br>
• 환율 {macro['exchange']}은 외국 자본 흐름과 투자 심리에 영향을 줍니다.<br>
• 물가 상승률 {macro['inflation']}%는 전체 임대료 수준을 끌어올립니다.<br>
• 수요 지수 {macro['demand']}는 해당 지역 선호도를 의미합니다.<br>
• 공급 지수 {macro['supply']}는 시장 포화도를 나타냅니다.<br><br>

<b>🧠 AI 해석:</b><br>

현재 시장은 단순한 부동산 요인이 아니라 거시경제 충격이 함께 작용하는 상태입니다.<br>
특히 금리와 공급 구조 변화가 가격 형성의 핵심 변수이며,<br>
단기적으로는 변동성이 높은 시장 환경입니다.<br><br>

따라서 해당 매물은 향후 경제 지표 변화에 따라 추가 상승 또는 조정 가능성이 모두 존재합니다.

</div>
"""

    st.markdown(explanation, unsafe_allow_html=True)

else:
    st.info("지도에서 매물을 클릭하면 분석이 시작됩니다.")
