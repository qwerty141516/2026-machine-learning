
import streamlit as st
import pandas as pd
import numpy as np
import random
import folium
import shap

from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(page_title="서울 대학가 원룸 미래 월세 예측 AI", layout="wide")
st.title("🏠 서울 대학가 원룸 미래 월세 예측 AI")

# ------------------ 데이터 생성 ------------------
random.seed(42)
np.random.seed(42)

data = []
universities = ["서울대", "연세대", "고려대"]

for i in range(3000):
    university = random.choice(universities)
    area = random.randint(10, 35)
    distance_to_univ = random.randint(0, 1000)
    building_age = random.randint(1, 30)
    floor = random.randint(1, 10)
    month = random.randint(1, 12)

    semester = 1 if month in [1,2,3,8,9] else 0

    base_rent = 55 if university=="연세대" else 50 if university=="고려대" else 48

    current_rent = max(
        base_rent + area*1.2 - distance_to_univ*0.015
        - building_age*0.5 + floor*0.8 + semester*6
        + np.random.normal(0,3),20
    )

    future_rent = current_rent + semester*2 + np.random.normal(1.5,2)

    data.append([university,area,distance_to_univ,building_age,
                 floor,month,semester,current_rent,future_rent])

df = pd.DataFrame(data, columns=[
    '대학교','면적','대학거리','건물연식','층수',
    '월','개강시즌','현재월세','미래월세'
])

df = pd.get_dummies(df, columns=['대학교'])

X = df[
    ['면적','대학거리','건물연식','층수','월',
     '개강시즌','현재월세',
     '대학교_고려대','대학교_서울대','대학교_연세대']
]
y = df['미래월세']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)

# ------------------ 대학 좌표 ------------------
UNIV = {
    "연세대": (37.5658, 126.9386),
    "서울대": (37.4599, 126.9519),
    "고려대": (37.5894, 127.0324)
}

# ------------------ 가짜 매물 ------------------
rooms = []
for univ, (lat, lon) in UNIV.items():
    for i in range(20):
        rooms.append({
            "매물명": f"{univ} 원룸 {i+1}",
            "대학교": univ,
            "lat": lat + random.uniform(-0.004, 0.004),
            "lon": lon + random.uniform(-0.004, 0.004),
            "면적": random.randint(10,35),
            "대학거리": random.randint(50,800),
            "건물연식": random.randint(1,25),
            "층수": random.randint(1,10),
            "현재월세": random.randint(35,90)
        })

rooms_df = pd.DataFrame(rooms)

selected_univ = st.selectbox("대학교 선택", ["연세대","서울대","고려대"])

filtered = rooms_df[rooms_df["대학교"] == selected_univ]
lat, lon = UNIV[selected_univ]

m = folium.Map(location=[lat, lon], zoom_start=15)

for _, row in filtered.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color="blue",
        fill=True,
        fill_opacity=0.7,
        popup=row["매물명"],
        tooltip=row["매물명"]
    ).add_to(m)

st.subheader("📍 대학 주변 매물")
map_data = st_folium(m, width=1000, height=500, returned_objects=["last_object_clicked"])

clicked_room = None

if map_data and map_data.get("last_object_clicked"):
    clat = map_data["last_object_clicked"]["lat"]
    clon = map_data["last_object_clicked"]["lng"]

    temp = filtered.copy()
    temp["dist"] = (temp["lat"]-clat)**2 + (temp["lon"]-clon)**2
    clicked_room = temp.sort_values("dist").iloc[0]

if clicked_room is not None:

    room = clicked_room

    with st.spinner("AI 분석 중..."):
        sample = pd.DataFrame({
            '면적':[room["면적"]],
            '대학거리':[room["대학거리"]],
            '건물연식':[room["건물연식"]],
            '층수':[room["층수"]],
            '월':[8],
            '개강시즌':[1],
            '현재월세':[room["현재월세"]],
            '대학교_고려대':[1 if selected_univ=="고려대" else 0],
            '대학교_서울대':[1 if selected_univ=="서울대" else 0],
            '대학교_연세대':[1 if selected_univ=="연세대" else 0]
        })

        future_rent = float(model.predict(sample)[0])
        change = ((future_rent-room["현재월세"])/room["현재월세"])*100

    st.subheader("🏢 선택한 매물")

    c1, c2 = st.columns(2)

    with c1:
        st.write(room["매물명"])
        st.write(f"현재 월세: {room['현재월세']}만원")
        st.write(f"면적: {room['면적']}㎡")
        st.write(f"층수: {room['층수']}층")
        st.write(f"건물연식: {room['건물연식']}년")
        st.write(f"거리: {room['대학거리']}m")

    with c2:
        st.metric("예측 월세", f"{future_rent:.1f}만원", f"{change:.1f}%")

        st.markdown("### 📌 SHAP 기반 예측 이유")
        shap_values = explainer.shap_values(sample)

        shap_df = pd.DataFrame({
            "feature": X.columns,
            "shap": shap_values[0],
            "value": sample.iloc[0].values
        }).sort_values("shap", ascending=False)

        for _, r in shap_df.head(5).iterrows():
            direction = "상승 요인" if r["shap"] > 0 else "하락 요인"
            st.write(f"- {r['feature']} ({direction})")

    st.markdown("### 📊 주변 시세 비교")

    avg = filtered["현재월세"].mean()
    diff = ((room["현재월세"] - avg)/avg)*100

    col1, col2 = st.columns(2)
    col1.metric("주변 평균", f"{avg:.1f}만원")
    col2.metric("시장 대비", f"{diff:.1f}%", "비쌈" if diff>0 else "저렴")

    st.markdown("### 📈 12개월 예측")

    months = list(range(1,13))
    preds = []

    base = sample.copy()

    for m in months:
        t = base.copy()
        t["월"] = m
        t["개강시즌"] = 1 if m in [1,2,3,8,9] else 0
        preds.append(float(model.predict(t)[0]))

    chart_df = pd.DataFrame({"month": months, "rent": preds})
    st.line_chart(chart_df.set_index("month"))
