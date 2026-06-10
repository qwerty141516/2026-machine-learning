import streamlit as st
import pandas as pd
import numpy as np
import random
import folium

from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# =========================
# 🎨 UI SETUP
# =========================
st.set_page_config(page_title="Zigbang AI", layout="wide")

st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: 800;
}
.sub {
    color: gray;
    margin-bottom: 10px;
}

.card {
    background: #111827;
    padding: 12px;
    border-radius: 12px;
    color: white;
    margin-bottom: 8px;
}

.price {
    font-size: 18px;
    font-weight: bold;
    color: #60a5fa;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏠 Zigbang AI - Real Estate Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>서울 대학가 원룸 매물 + 투자 분석 AI</div>", unsafe_allow_html=True)

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

        future = rent + semester*2 + np.random.normal(1.5,2)

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
        for i in range(30):
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
# 🎯 FILTER
# =========================
selected = st.selectbox("🏫 대학 선택", list(UNIV.keys()))
filtered = rooms[rooms["univ"] == selected]

# =========================
# 📐 LAYOUT (MAP + SIDE PANEL)
# =========================
col1, col2 = st.columns([1.6, 1])

# =========================
# 🗺️ MAP
# =========================
with col1:
    st.subheader("🗺️ 매물 지도 (직방 스타일)")

    lat, lon = UNIV[selected]
    m = folium.Map(location=[lat, lon], zoom_start=15)

    for _, r in filtered.iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]],
            radius=5,
            fill=True,
            tooltip=r["name"]
        ).add_to(m)

    map_data = st_folium(m, height=650)

# =========================
# 📋 RIGHT PANEL
# =========================
with col2:

    st.subheader("🏠 매물 리스트")

    for _, r in filtered.head(6).iterrows():
        st.markdown(f"""
        <div class="card">
            <b>{r['name']}</b><br>
            <div class="price">{r['rent']}만원</div>
            <small>
            {r['area']}㎡ · {r['floor']}층 · {r['age']}년
            </small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🧠 AI 분석")

    clicked = None

    if map_data and map_data.get("last_object_clicked"):
        c = map_data["last_object_clicked"]

        temp = filtered.copy()
        temp["d"] = (temp["lat"]-c["lat"])**2 + (temp["lon"]-c["lng"])**2
        clicked = temp.sort_values("d").iloc[0]

    if clicked is not None:

        # =========================
        # PREDICTION
        # =========================
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

        pred = float(model.predict(sample)[0])
        change = (pred - clicked["rent"]) / clicked["rent"] * 100

        st.metric("📈 미래 월세", f"{pred:.1f}만원", f"{change:.1f}%")

        # =========================
        # 🧠 INVESTMENT SCORE
        # =========================
        score = 50

        if clicked["dist"] < 300: score += 20
        if clicked["age"] < 5: score += 15
        if clicked["area"] > 25: score += 10
        if change > 5: score += 10
        if change < 0: score -= 10

        st.metric("🧠 투자 점수", f"{score}/100")

        if score >= 80:
            st.success("🔥 적극 추천 매물")
        elif score >= 60:
            st.info("🟡 조건부 추천")
        else:
            st.warning("🔴 비추천")

        # =========================
        # 📊 MARKET TREND
        # =========================
        st.markdown("### 📈 서울 월세 상승 그래프")

        years = list(range(2018, 2026))
        trend = [50 + i*3 + random.uniform(-1,1) for i in range(len(years))]

        st.line_chart(pd.DataFrame({
            "year": years,
            "rent": trend
        }).set_index("year"))

        # =========================
        # 🏠 SIMILAR PROPERTIES
        # =========================
        st.markdown("### 🏠 비슷한 매물")

        similar = filtered.copy()
        similar["diff"] = abs(similar["rent"] - clicked["rent"])
        similar = similar.sort_values("diff").head(3)

        for _, r in similar.iterrows():
            st.write(f"• {r['name']} / {r['rent']}만원 / {r['dist']}m")

        # =========================
        # ⏳ BUY OR WAIT
        # =========================
        st.markdown("### ⏳ 투자 판단")

        if change > 5 and score > 70:
            st.success("👉 지금 매수 추천")
            st.write("📌 상승 추세 + 입지 우수 + 수요 증가")
        elif change > 0:
            st.info("👉 관망 또는 빠른 결정")
            st.write("📌 완만한 상승 예상")
        else:
            st.warning("👉 기다리는 것이 유리")
            st.write("📌 단기 하락 또는 과대평가")

    else:
        st.info("지도에서 매물을 클릭하면 분석이 시작됩니다")
