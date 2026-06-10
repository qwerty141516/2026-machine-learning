import streamlit as st
import pandas as pd
import numpy as np
import random
import folium

from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# =========================
# 🎨 UI SETUP (CENTERED STYLE)
# =========================
st.set_page_config(page_title="Seoul Rent AI", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1100px;
    margin: auto;
}

.title {
    font-size: 34px;
    font-weight: 800;
    text-align: center;
}

.sub {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}

.card {
    background:#111827;
    padding:15px;
    border-radius:14px;
    color:white;
    margin-top:10px;
}

.big {
    font-size: 22px;
    font-weight: 700;
    color: #60a5fa;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🏠 Seoul Real Estate AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>AI 기반 미래 월세 예측 & 투자 분석</div>", unsafe_allow_html=True)

# =========================
# 🏫 DATA
# =========================
UNIV = {
    "연세대": (37.5658, 126.9386),
    "서울대": (37.4599, 126.9519),
    "고려대": (37.5894, 127.0324)
}

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
# 🗺️ MAP CENTER
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
# 📌 CENTER ANALYSIS PANEL
# =========================
st.markdown("---")

clicked = None

if map_data and map_data.get("last_object_clicked"):
    c = map_data["last_object_clicked"]

    temp = filtered.copy()
    temp["d"] = (temp["lat"]-c["lat"])**2 + (temp["lon"]-c["lng"])**2
    clicked = temp.sort_values("d").iloc[0]

# =========================
# 🧠 ANALYSIS
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

    pred = float(model.predict(sample)[0])
    change = (pred - clicked["rent"]) / clicked["rent"] * 100

    # =========================
    # 💰 RESULT CARD
    # =========================
    st.markdown(f"""
    <div class="card">
        <div class="big">📈 미래 월세: {pred:.1f}만원</div>
        현재 {clicked['rent']}만원 → 변화 {change:.1f}%
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 🧠 GPT STYLE EXPLANATION
    # =========================
    st.markdown("### 🧠 AI 분석 리포트 (GPT 스타일)")

    def gpt_style_text(r, p, ch):
        text = f"""
이 매물은 현재 {r['rent']}만원 수준에서 형성되어 있으며,
AI 예측 결과 약 {p:.1f}만원까지 변화할 것으로 분석됩니다.

이 변화의 핵심 요인은 다음과 같습니다.

- 해당 매물은 대학과의 거리 {r['dist']}m 수준으로 수요 영향이 반영됩니다.
- 건물 연식은 {r['age']}년으로, 노후도에 따라 가격 압력이 존재합니다.
- 면적 {r['area']}㎡는 수요 대비 경쟁력을 결정하는 요소입니다.
- 시장 전체적으로는 학기 시즌 및 지역 수요 변화가 함께 반영됩니다.

종합적으로 판단하면,
"""
        if ch > 5:
            text += "상승 압력이 강한 지역으로, 단기적으로도 추가 상승 가능성이 존재합니다."
        elif ch > 0:
            text += "완만한 상승 흐름이 예상되는 안정적인 매물입니다."
        else:
            text += "단기적으로는 가격 조정 또는 횡보 가능성이 존재합니다."

        return text

    st.info(gpt_style_text(clicked, pred, change))

    # =========================
    # 🧠 INVESTMENT SCORE
    # =========================
    score = 50
    if clicked["dist"] < 300: score += 20
    if clicked["age"] < 5: score += 15
    if change > 5: score += 10
    if change < 0: score -= 10

    st.metric("🧠 투자 점수", f"{score}/100")

    if score >= 80:
        st.success("🔥 적극 추천")
    elif score >= 60:
        st.info("🟡 조건부 추천")
    else:
        st.warning("🔴 비추천")

    # =========================
    # ⏳ BUY / WAIT
    # =========================
    st.markdown("### ⏳ 투자 판단")

    if change > 5 and score > 70:
        st.success("👉 지금 매수 추천 (상승 모멘텀 존재)")
    elif change > 0:
        st.info("👉 관망 또는 빠른 결정 필요")
    else:
        st.warning("👉 기다리는 것이 유리")

else:
    st.info("지도에서 매물을 클릭하면 AI 분석이 시작됩니다")
