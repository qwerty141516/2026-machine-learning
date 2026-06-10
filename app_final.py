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

# =========================
# 🏷️ TITLE
# =========================
st.markdown("<div class='title'>🏠 서울 주요 대학 근처 월세 예측</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>금리·수요·공급 기반 AI 월세 시뮬레이션</div>", unsafe_allow_html=True)

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
# 🎯 UNIVERSITY SELECT
# =========================
selected = st.selectbox("🏫 대학 선택", list(UNIV.keys()))

filtered = rooms[rooms["univ"] == selected]

# =========================
# 🗺️ MAP (선택 강조)
# =========================
lat, lon = UNIV[selected]
m = folium.Map(location=[lat, lon], zoom_start=15)

map_data = st_folium(m, height=600, width=1100)

# =========================
# 📌 CLICK DETECTION
# =========================
clicked = None

if map_data and map_data.get("last_object_clicked"):

    with st.spinner("🧠 AI가 해당 매물을 분석 중입니다..."):
        c = map_data["last_object_clicked"]

        temp = filtered.copy()
        temp["d"] = (temp["lat"]-c["lat"])**2 + (temp["lon"]-c["lng"])**2
        clicked = temp.sort_values("d").iloc[0]

# =========================
# 🌍 ECONOMIC MODEL
# =========================
def market_shift(month, age, dist):
    shift = 0

    shift -= 1.8  # 금리

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

    shift += 1.0  # 인플레이션
    return shift

# =========================
# 📍 ANALYSIS SECTION
# =========================
st.markdown("---")
st.markdown("<div id='analysis'></div>", unsafe_allow_html=True)

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
이 매물은 단순히 가격이 오르는/내리는 구조가 아니라,
**입지 + 수요 시즌 + 금리 + 노후도 + 거리 변수**가 결합된 결과입니다.

---

📍 현재 월세: **{clicked['rent']}만원**  
📊 AI 예측 미래 월세: **{future_price:.1f}만원**

---

### 🔍 가격 변화 해석

이 매물의 가격 변화는 다음과 같이 해석됩니다:

- 💰 금리 효과: 기준 금리 상승 압력으로 -1.8%
- 🎓 계절 수요: 개강 시즌 영향으로 +수요 폭발 (+3.2%)
- 🏗️ 노후도: 건물 상태에 따라 ± 영향 반영
- 🚶 거리 효과: 역/대학 접근성이 핵심 변수
- 📈 시장 인플레이션: 전체적으로 +1.0% 상승 압력

---

### ⚖️ 주변 대비 위치

- 이 매물 평균 월세: **{avg_rent:.1f}만원**
- 시장 대비 차이: **{((clicked['rent']-avg_rent)/avg_rent*100):.1f}%**

👉 즉, 이 매물은 시장 평균 대비 
{"고가 프리미엄" if clicked['rent'] > avg_rent else "저평가 구간"}에 위치합니다.
""")

    # =========================
    # 📊 RESULT CARD
    # =========================
    st.markdown(f"""
    <div class="card">
        📍 미래 월세: <b>{future_price:.1f}만원</b><br>
        변화율: {change:.1f}%
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 📊 MARKET STATE
    # =========================
    total_shift = shift

    st.markdown("### 📊 시장 상태")

    if total_shift > 3:
        st.success("📈 강한 상승 시장")
    elif total_shift > 0:
        st.info("📊 완만한 상승 시장")
    else:
        st.warning("📉 조정 또는 하락 시장")

    # =========================
    # 🧠 SCORE
    # =========================
    score = 50
    if clicked["dist"] < 300: score += 20
    if clicked["age"] < 5: score += 15
    if change > 5: score += 10
    if change < 0: score -= 10

    st.metric("🧠 투자 점수", f"{score}/100")

    # =========================
    # 🎯 DECISION
    # =========================
    st.markdown("### ⏳ 투자 판단")

    if change > 5 and score > 70:
        st.success("👉 지금 매수 (상승 사이클)")
    elif change > 0:
        st.info("👉 보유 또는 관망")
    else:
        st.warning("👉 진입 보류")

else:
    st.info("지도에서 매물을 클릭하면 AI 분석이 시작됩니다.")
