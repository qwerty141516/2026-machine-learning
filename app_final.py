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
st.set_page_config(page_title="Seoul Rent Economic Simulator", layout="wide")

st.markdown("""
<style>
.block-container {
    max-width: 1100px;
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

st.markdown("<div class='title'>🏠 Seoul Rent Economic Simulator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>금리·수요·공급 기반 실제 월세 변동 시뮬레이션</div>", unsafe_allow_html=True)

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
def make_data():
    random.seed(42)
    np.random.seed(42)

    data = []

    for _ in range(3000):
        u = random.choice(list(UNIV.keys()))
        area = random.randint(10, 35)
        dist = random.randint(0, 1000)
        age = random.randint(1, 30)
        floor = random.randint(1, 10)
        month = random.randint(1, 12)

        semester = 1 if month in [1,2,3,8,9] else 0
        base = 55 if u=="연세대" else 50 if u=="고려대" else 48

        rent = base + area*1.2 - dist*0.015 - age*0.5 + floor*0.8 + semester*6
        rent += np.random.normal(0,3)

        future = rent + np.random.normal(1.5,2)

        data.append([u, area, dist, age, floor, month, semester, rent, future])

    df = pd.DataFrame(data, columns=[
        "대학교","면적","거리","연식","층","월","개강","현재","미래"
    ])

    return pd.get_dummies(df, columns=["대학교"])

df = make_data()

X = df.drop(columns=["미래"])
y = df["미래"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# =========================
# 🤖 MODEL
# =========================
@st.cache_resource
def train_model():
    model = XGBRegressor(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

# =========================
# 🏠 ROOMS
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

# =========================
# 🎯 UI
# =========================
selected = st.selectbox("🏫 대학 선택", list(UNIV.keys()))
filtered = rooms[rooms["univ"] == selected]

# =========================
# 🗺️ MAP
# =========================
lat, lon = UNIV[selected]
m = folium.Map(location=[lat, lon], zoom_start=15)

for _, r in filtered.iterrows():
    folium.CircleMarker(
        [r["lat"], r["lon"]],
        radius=5,
        fill=True,
        tooltip=r["name"]
    ).add_to(m)

map_data = st_folium(m, height=600)

# =========================
# 🌍 ECONOMIC ENGINE (핵심)
# =========================
def market_shift(month, age, dist):

    shift = 0

    # 💰 금리 영향 (현재 상승 국면 가정)
    interest_rate = 1.8
    shift -= interest_rate

    # 🎓 수요 시즌
    if month in [2,3,8,9]:
        shift += 3.2
    else:
        shift -= 1.2

    # 🏗️ 공급 효과
    if age < 5:
        shift -= 1.5
    elif age > 15:
        shift += 1.2

    # 🚶 입지 수요
    if dist < 300:
        shift += 2.5
    elif dist > 700:
        shift -= 1

    # 💰 인플레이션
    shift += 1.0

    return shift

# =========================
# 📌 ANALYSIS
# =========================
st.markdown("---")

clicked = None

if map_data and map_data.get("last_object_clicked"):
    c = map_data["last_object_clicked"]

    temp = filtered.copy()
    temp["d"] = (temp["lat"]-c["lat"])**2 + (temp["lon"]-c["lng"])**2
    clicked = temp.sort_values("d").iloc[0]

# =========================
# 🧠 RESULT
# =========================
if clicked is not None:

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

    base_price = model.predict(sample)[0]

    shift = market_shift(8, clicked["age"], clicked["dist"])

    future_price = base_price * (1 + shift/100)

    change = (future_price - clicked["rent"]) / clicked["rent"] * 100

    # =========================
    # 💰 RESULT CARD
    # =========================
    st.markdown(f"""
    <div class="card">
        📍 미래 월세: <b>{future_price:.1f}만원</b><br>
        현재 → 변화: {change:.1f}%
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 🌍 ECONOMIC BREAKDOWN
    # =========================
    st.markdown("### 🌍 경제 변수 분석")

    st.write(f"💰 금리 영향: -1.8%")
    st.write(f"🎓 수요 시즌 영향: +3.2%")
    st.write(f"🏗️ 공급 영향: 변동 반영")
    st.write(f"🚶 입지 영향: 자동 반영")
    st.write(f"📈 인플레이션: +1.0%")

    total_shift = shift

    st.markdown("### 📊 총 시장 변화")

    if total_shift > 3:
        st.success("📈 강한 상승 시장")
    elif total_shift > 0:
        st.info("📊 완만한 상승 시장")
    else:
        st.warning("📉 하락 또는 조정 시장")

    # =========================
    # 🧠 INVESTMENT SCORE
    # =========================
    score = 50
    if clicked["dist"] < 300: score += 20
    if clicked["age"] < 5: score += 15
    if change > 5: score += 10
    if change < 0: score -= 10

    st.metric("🧠 투자 점수", f"{score}/100")

    # =========================
    # ⏳ DECISION
    # =========================
    st.markdown("### ⏳ 투자 판단")

    if change > 5 and score > 70:
        st.success("👉 지금 매수 (상승 사이클 진입)")
    elif change > 0:
        st.info("👉 보유 또는 관망")
    else:
        st.warning("👉 대기 전략 추천")

else:
    st.info("지도에서 매물을 클릭하면 경제 시뮬레이션이 시작됩니다")
