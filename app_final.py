import streamlit as st
import pandas as pd
import numpy as np
import random
import folium
import shap
import matplotlib.pyplot as plt

from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ------------------ UI 기본 설정 ------------------
st.set_page_config(
    page_title="Seoul Rent AI",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 16px;
        color: gray;
        margin-bottom: 30px;
    }
    .card {
        background: #111827;
        padding: 20px;
        border-radius: 12px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏠 Seoul Rent AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>대학가 원룸 월세 예측 & 분석 플랫폼</div>", unsafe_allow_html=True)

# ------------------ 데이터 생성 ------------------
@st.cache_data
def make_data():
    random.seed(42)
    np.random.seed(42)

    data = []
    universities = ["서울대", "연세대", "고려대"]

    for i in range(3000):
        univ = random.choice(universities)
        area = random.randint(10, 35)
        dist = random.randint(0, 1000)
        age = random.randint(1, 30)
        floor = random.randint(1, 10)
        month = random.randint(1, 12)

        semester = 1 if month in [1,2,3,8,9] else 0
        base = 55 if univ=="연세대" else 50 if univ=="고려대" else 48

        current = max(
            base + area*1.2 - dist*0.015
            - age*0.5 + floor*0.8 + semester*6
            + np.random.normal(0,3),
            20
        )

        future = current + semester*2 + np.random.normal(1.5,2)

        data.append([univ, area, dist, age, floor, month, semester, current, future])

    df = pd.DataFrame(data, columns=[
        "대학교","면적","거리","연식","층","월","개강","현재","미래"
    ])

    df = pd.get_dummies(df, columns=["대학교"])
    return df


df = make_data()

X = df.drop(columns=["미래"])
y = df["미래"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ------------------ 모델 ------------------
@st.cache_resource
def train_model():
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

@st.cache_resource
def get_explainer(model):
    return shap.TreeExplainer(model)

explainer = get_explainer(model)

# ------------------ 대학 위치 ------------------
UNIV = {
    "연세대": (37.5658, 126.9386),
    "서울대": (37.4599, 126.9519),
    "고려대": (37.5894, 127.0324)
}

# ------------------ 매물 생성 ------------------
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

# ------------------ 사이드바 ------------------
st.sidebar.header("🎯 필터")
selected = st.sidebar.selectbox("대학교 선택", list(UNIV.keys()))

filtered = rooms[rooms["univ"] == selected]

# ------------------ 레이아웃 ------------------
col1, col2 = st.columns([1.2, 1])

# ================= LEFT: MAP =================
with col1:
    st.subheader("📍 매물 지도")

    lat, lon = UNIV[selected]
    m = folium.Map(location=[lat, lon], zoom_start=15)

    for _, r in filtered.iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]],
            radius=5,
            fill=True,
            popup=r["name"]
        ).add_to(m)

    map_data = st_folium(m, height=550)

# ================= RIGHT: INFO =================
with col2:

    st.subheader("📊 분석 패널")

    clicked = None

    if map_data and map_data.get("last_object_clicked"):
        c = map_data["last_object_clicked"]

        temp = filtered.copy()
        temp["d"] = (temp["lat"]-c["lat"])**2 + (temp["lon"]-c["lng"])**2
        clicked = temp.sort_values("d").iloc[0]

    if clicked is not None:

        st.markdown("### 🏢 선택 매물")

        st.markdown(f"""
        <div class="card">
        <h4>{clicked['name']}</h4>
        <p>현재 월세: {clicked['rent']}만원</p>
        <p>면적: {clicked['area']}㎡</p>
        <p>층수: {clicked['floor']}층</p>
        <p>연식: {clicked['age']}년</p>
        </div>
        """, unsafe_allow_html=True)

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

        st.metric("📈 예측 월세", f"{pred:.1f}만원", f"{change:.1f}%")

        # ---------------- SHAP ----------------
        st.markdown("### 🧠 AI 분석")

        shap_values = explainer.shap_values(sample)[0]

        fig, ax = plt.subplots()
        ax.barh(X.columns, shap_values)
        ax.set_title("Feature Impact (SHAP)")

        st.pyplot(fig)

        # ---------------- 시장 비교 ----------------
        st.markdown("### 📊 시장 비교")

        avg = filtered["rent"].mean()

        c1, c2 = st.columns(2)
        c1.metric("평균 월세", f"{avg:.1f}만원")
        c2.metric("시장 대비", f"{((clicked['rent']-avg)/avg)*100:.1f}%")

        # ---------------- 예측 그래프 ----------------
        st.markdown("### 📈 12개월 전망")

        months = range(1,13)
        preds = []

        for m in months:
            s = sample.copy()
            s["월"] = m
            s["개강"] = 1 if m in [1,2,3,8,9] else 0
            preds.append(float(model.predict(s)[0]))

        st.line_chart(pd.DataFrame({"month": months, "rent": preds}).set_index("month"))

    else:
        st.info("지도에서 매물을 클릭하면 분석이 시작됩니다")
