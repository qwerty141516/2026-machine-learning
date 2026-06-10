import streamlit as st
import pandas as pd
import numpy as np
import random
import folium

from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# =========================
# 🎨 UI
# =========================
st.set_page_config(page_title="서울 주요 대학 근처 월세 예측", layout="wide")

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
st.markdown("<div class='sub'>AI 기반 월세 변동 시뮬레이션</div>", unsafe_allow_html=True)

# =========================
# 🏫 UNIVERSITY
# =========================
UNIV = {
    "연세대": (37.5658, 126.9386),
    "서울대": (37.4599, 126.9519),
    "고려대": (37.5894, 127.0324)
}

# =========================
# 📊 DATA
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
                "area": random.randint(10,35),
                "dist": random.randint(50,800),
                "age": random.randint(1,25),
                "floor": random.randint(1,10),
                "rent": random.randint(35,90)
            })
    return pd.DataFrame(rooms)

rooms = make_rooms()

selected = st.selectbox("🏫 대학 선택", list(UNIV.keys()))
filtered = rooms[rooms["univ"] == selected]

# =========================
# 🗺️ MAP (핵심 수정: 매물 다시 표시)
# =========================
lat, lon = UNIV[selected]
m = folium.Map(location=[lat, lon], zoom_start=15)

map_data = st_folium(m, height=600, width=1100)

# =========================
# 📌 CLICK DETECTION
# =========================
clicked = None

if map_data and map_data.get("last_object_clicked"):
    c = map_data["last_object_clicked"]

    temp = filtered.copy()
    temp["d"] = (temp["lat"]-c["lat"])**2 + (temp["lon"]-c["lng"])**2
    clicked = temp.sort_values("d").iloc[0]

# =========================
# 🗺️ 다시 매물 그리기 (여기가 핵심)
# =========================
for _, r in filtered.iterrows():

    is_selected = False

    if clicked is not None:
        is_selected = (r["name"] == clicked["name"])

    if is_selected:
        color = "red"
        radius = 8
    else:
        color = "blue"
        radius = 4

    folium.CircleMarker(
        location=[r["lat"], r["lon"]],
        radius=radius,
        color=color,
        fill=True,
        fill_opacity=0.8,
        tooltip=f"{r['name']} | {r['rent']}만원"
    ).add_to(m)

# 다시 렌더 (중요)
map_data = st_folium(m, height=600, width=1100)

# =========================
# 🌍 ECONOMY
# =========================
def market_shift(month, age, dist):
    shift = -1.8

    if month in [2,3,8,9]:
        shift += 3.2
    else:
        shift -= 1.2

    if age < 5:
        shift -= 1.5
    elif age > 15:
        shift += 1.2

    if dist < 300:
        shift += 2.5
    elif dist > 700:
        shift -= 1

    shift += 1.0
    return shift

# =========================
# 📊 ANALYSIS
# =========================
st.markdown("---")
st.markdown("<div id='analysis'></div>", unsafe_allow_html=True)

if clicked is not None:

    with st.spinner("🧠 AI가 매물 데이터를 분석 중입니다..."):
        sample = pd.DataFrame({
            "면적":[clicked["area"]],
            "거리":[clicked["dist"]],
            "연식":[clicked["age"]],
            "층":[clicked["floor"]],
            "월":[8],
            "개강":[1],
            "현재":[clicked["rent"]],
            "대학교_고려대":[1 if selected=="고려대" else 0],
            "대학교_서울대":[1 if selected=="서울대" else 0],
            "대학교_연세대":[1 if selected=="연세대" else 0]
        })

        base_price = clicked["rent"] * 1.02
        shift = market_shift(8, clicked["age"], clicked["dist"])
        future_price = base_price * (1 + shift/100)

        change = (future_price - clicked["rent"]) / clicked["rent"] * 100

        avg_rent = filtered["rent"].mean()

    # 자동 스크롤
    st.markdown("""
    <script>
        document.getElementById("analysis").scrollIntoView({behavior: "smooth"});
    </script>
    """, unsafe_allow_html=True)

    # =========================
    # 🤖 GPT 스타일 설명
    # =========================
    st.markdown("## 🤖 AI 분석 리포트")

    st.markdown(f"""
이 매물은 단순한 가격 변화가 아니라
**입지, 수요, 계절성, 노후도, 거리 변수**가 결합된 결과입니다.

---

📍 현재 월세: **{clicked['rent']}만원**  
📊 예상 미래 월세: **{future_price:.1f}만원**

---

### 🔍 가격 변화 구조

- 💰 금리 영향: -1.8%
- 🎓 계절 수요: +3.2%
- 🏗️ 노후도 영향 반영
- 🚶 거리 프리미엄 반영
- 📈 인플레이션 +1.0%

---

### ⚖️ 시장 비교

- 평균 월세: **{avg_rent:.1f}만원**
- 시장 대비: {"고평가" if clicked['rent'] > avg_rent else "저평가"}
""")

    st.markdown(f"""
    <div class="card">
        📍 미래 월세: <b>{future_price:.1f}만원</b><br>
        변화율: {change:.1f}%
    </div>
    """, unsafe_allow_html=True)

    score = 50
    if clicked["dist"] < 300: score += 20
    if clicked["age"] < 5: score += 15
    if change > 5: score += 10
    if change < 0: score -= 10

    st.metric("🧠 투자 점수", f"{score}/100")

else:
    st.info("지도에서 매물을 클릭하면 분석이 시작됩니다.")
