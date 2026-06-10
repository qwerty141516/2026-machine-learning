import streamlit as st
import pandas as pd
import numpy as np
import random
import folium

from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ------------------ 설정 ------------------
st.set_page_config(page_title="Seoul Rent AI", layout="wide")
st.title("🏠 Seoul Rent AI (Fast Mode)")

UNIV = {
    "연세대": (37.5658, 126.9386),
    "서울대": (37.4599, 126.9519),
    "고려대": (37.5894, 127.0324)
}

# ------------------ 데이터 (캐시) ------------------
@st.cache_data
def make_data():
    random.seed(42)
    np.random.seed(42)

    data = []
    universities = list(UNIV.keys())

    for i in range(3000):
        u = random.choice(universities)
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

# ------------------ 모델 (1번만 생성) ------------------
@st.cache_resource
def train_model():
    model = XGBRegressor(
        n_estimators=120,   # 🔥 속도 핵심 (300 → 120)
        learning_rate=0.08,
        max_depth=5,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

# ------------------ 매물 ------------------
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

# ------------------ UI ------------------
selected = st.selectbox("대학교 선택", list(UNIV.keys()))
filtered = rooms[rooms["univ"] == selected]

# ------------------ 지도 ------------------
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

# ------------------ 클릭 처리 ------------------
clicked = None

if map_data and map_data.get("last_object_clicked"):
    c = map_data["last_object_clicked"]

    temp = filtered.copy()
    temp["d"] = (temp["lat"]-c["lat"])**2 + (temp["lon"]-c["lng"])**2
    clicked = temp.sort_values("d").iloc[0]

# ------------------ 예측 ------------------
if clicked is not None:

    st.subheader("🏢 선택 매물")

    st.write(clicked["name"])
    st.write(f"현재 월세: {clicked['rent']}만원")
    st.write(f"면적: {clicked['area']}㎡")
    st.write(f"층수: {clicked['floor']}층")

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

    st.markdown("### 📊 시장 비교")

    avg = filtered["rent"].mean()

    col1, col2 = st.columns(2)
    col1.metric("평균", f"{avg:.1f}만원")
    col2.metric("차이", f"{((clicked['rent']-avg)/avg)*100:.1f}%")

else:
    st.info("지도에서 매물을 클릭하세요")
