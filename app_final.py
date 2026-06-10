import streamlit as st
import pandas as pd
import numpy as np
import random
import folium
import time
import matplotlib.pyplot as plt

from streamlit_folium import st_folium

# =========================
# 🌐 CONFIG
# =========================
st.set_page_config(page_title="AI Rent Simulator", layout="wide")

st.markdown("""
<h1 style='text-align:center;'>🏠 AI Rent + Macro Economy Simulator</h1>
<p style='text-align:center;color:gray;'>지도 + 경제 변화 기반 월세 분석</p>
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
                "area": random.randint(10, 40),
                "dist": random.randint(50, 900),
                "age": random.randint(1, 25),
                "floor": random.randint(1, 10),
                "rent": random.randint(35, 95)
            })
    return pd.DataFrame(rooms)

rooms = make_rooms()

# =========================
# 🏦 MACRO ECONOMY (자동 시뮬레이션)
# =========================
macro = {
    "interest": random.uniform(2, 5),
    "inflation": random.uniform(1.5, 4),
    "exchange": random.uniform(1200, 1500),
    "event": random.uniform(-1, 1),
    "supply": random.uniform(0.6, 1.4)
}

# =========================
# 🧠 PRICE MODEL
# =========================
def predict_rent(base, macro):
    return max(30,
        base
        + macro["interest"] * 4
        + (macro["exchange"] - 1300) * 0.03
        + macro["inflation"] * 3
        + macro["event"] * 10
        - macro["supply"] * 12
    )

# =========================
# 📌 UI LAYOUT FIX (핵심)
# =========================
col1, col2 = st.columns([1, 2])

with col1:
    selected = st.selectbox("🏫 대학 선택", list(UNIV.keys()))

    st.markdown("## 📊 경제 상황 (자동)")
    st.write(f"금리: {macro['interest']:.2f}%")
    st.write(f"물가: {macro['inflation']:.2f}%")
    st.write(f"환율: {macro['exchange']:.0f}")
    st.write(f"지역 이벤트: {macro['event']:.2f}")
    st.write(f"공급지수: {macro['supply']:.2f}")

filtered = rooms[rooms["univ"] == selected]
lat, lon = UNIV[selected]

# =========================
# 🗺 MAP (오른쪽 고정)
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

    map_data = st_folium(m, height=650, width=700)

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

    base_rent = clicked["rent"]
    pred = predict_rent(base_rent, macro)

    # =========================
    # 📊 PRICE ANIMATION
    # =========================
    st.markdown("## 📈 가격 변화 시뮬레이션")

    history = np.linspace(base_rent, pred, 12)

    chart = st.empty()

    for i in range(2, len(history)):
        fig, ax = plt.subplots()
        ax.plot(history[:i], marker="o")
        ax.set_ylim(min(history)-5, max(history)+5)
        ax.set_title("월세 변화 (경제 충격 반영)")
        chart.pyplot(fig)
        time.sleep(0.1)

    # =========================
    # 💰 RESULT
    # =========================
    colA, colB = st.columns(2)

    with colA:
        st.metric("현재 월세", f"{base_rent}만원")

    with colB:
        st.metric("예측 월세", f"{pred:.1f}만원")

    # =========================
    # 🤖 GPT STYLE EXPLANATION
    # =========================
    st.markdown("## 🤖 AI 설명")

    reasons = []

    if macro["interest"] > 4:
        reasons.append("금리 상승 → 대출 부담 증가 → 월세 상승 압력")

    if macro["inflation"] > 3:
        reasons.append("물가 상승 → 전체 임대료 상승 구조")

    if macro["exchange"] > 1400:
        reasons.append("환율 상승 → 외국인 수요 증가 가능성")

    if macro["event"] > 0.3:
        reasons.append("지역 호재 → 수요 증가")

    if macro["event"] < -0.3:
        reasons.append("지역 악재 → 수요 감소")

    if macro["supply"] < 0.8:
        reasons.append("공급 부족 → 가격 상승 압력")

    if macro["supply"] > 1.2:
        reasons.append("공급 과잉 → 가격 하락 압력")

    st.markdown(f"""
    ### 🧠 AI 분석 결과

    선택 매물 기준:
    - 현재 월세: **{base_rent}만원**
    - 예측 월세: **{pred:.1f}만원**

    **왜 이렇게 변했나?**
    """)

    for r in reasons:
        st.write("• " + r)

else:
    st.info("지도에서 매물을 클릭하면 분석이 시작됩니다.")
