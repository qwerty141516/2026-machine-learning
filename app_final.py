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

# 🔧 글자 깨짐 방지 (핵심)
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
}

/* 그래프 너무 커지는 것 방지 */
.stPlotlyChart, .stPyplot {
    max-width: 700px;
    margin: auto;
}

/* AI 박스 */
.ai-box {
    background: linear-gradient(135deg, #111827, #1f2937);
    padding: 20px;
    border-radius: 16px;
    color: white;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🏠 AI Rent Simulator</h1>", unsafe_allow_html=True)

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
# 🏫 UI (정렬 유지)
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
            fill_opacity=0.8
        ).add_to(m)

    map_data = st_folium(m, height=600)

# =========================
# 📌 CLICK
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

    # 간단 경제 시뮬레이션
    macro_score = random.uniform(-1, 1)
    pred = base + macro_score * 15

    # =========================
    # 📈 SMALLER GRAPH (핵심 수정)
    # =========================
    st.markdown("### 📊 가격 변화")

    history = np.linspace(base, pred, 10)

    fig, ax = plt.subplots(figsize=(6, 3))  # 👈 크기 줄임
    ax.plot(history, marker="o")
    ax.set_title("월세 변화 시뮬레이션")
    ax.grid(True)

    st.pyplot(fig)

    # =========================
    # 💰 RESULT
    # =========================
    colA, colB = st.columns(2)

    with colA:
        st.metric("현재 월세", f"{base}만원")

    with colB:
        st.metric("예측 월세", f"{pred:.1f}만원")

    # =========================
    # 🤖 GPT STYLE LONG EXPLANATION (핵심)
    # =========================
    st.markdown("## 🤖 AI 분석 리포트")

    explanation = f"""
<div class="ai-box">

현재 선택된 매물은 {selected} 인근 지역에 위치한 원룸이며,
현재 월세는 <b>{base}만원</b>, 경제 상황을 반영한 예상 월세는 <b>{pred:.1f}만원</b>으로 분석됩니다.

<br><br>

이 변화는 단순한 부동산 요인이 아니라, 현재 시장에 영향을 주는 거시경제 흐름에 의해 발생합니다.

<br><br>

<b>📌 주요 분석 내용:</b><br>

- 최근 경제 변동성에 따라 임대 시장 전반의 가격 민감도가 증가하고 있습니다.<br>
- 지역 내 수요 대비 공급 수준 변화가 가격 형성에 직접적인 영향을 주고 있습니다.<br>
- 투자 심리와 금리 환경 변화는 월세 가격 상승 압력 또는 하락 압력으로 작용합니다.<br>

<br>

<b>🧠 AI 해석:</b><br>
현재 시장은 안정적이지 않으며, 단기적으로는 변동성이 존재하는 상태입니다.
따라서 해당 매물의 가격은 향후 경제 지표 변화에 따라 추가 상승 또는 조정 가능성이 있습니다.

</div>
"""

    st.markdown(explanation, unsafe_allow_html=True)

else:
    st.info("지도에서 매물을 클릭하면 AI 분석이 시작됩니다.")
