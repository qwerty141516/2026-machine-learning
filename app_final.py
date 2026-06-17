import streamlit as st
import pandas as pd
import random
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# 한글 폰트 설정 (Matplotlib 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows용
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="서울 대학가 원룸 월세 예측 모델", layout="wide")

st.markdown("""
<h1 style='text-align:center;'>🏠 서울 대학가 원룸 월세 예측 모델</h1>
<p style='text-align:center;color:gray;'>AI 기반 거시경제·전세/월세 분석 시스템</p>
""", unsafe_allow_html=True)

# =========================
# 대학 좌표
# =========================
UNIV = {
    "연세대": (37.5658, 126.9386),
    "서울대": (37.4599, 126.9519),
    "고려대": (37.5894, 127.0324)
}

# =========================
# 가상 매물 생성
# =========================
@st.cache_data
def make_rooms():
    rows = []
    random.seed(42)

    for u, (lat, lon) in UNIV.items():
        for i in range(30):
            deposit = random.randint(500, 3000)
            rent = random.randint(40, 90)

            rows.append({
                "name": f"{u} 원룸 {i+1}",
                "univ": u,
                "lat": lat + random.uniform(-0.004, 0.004),
                "lon": lon + random.uniform(-0.004, 0.004),

                "rent": rent,
                "deposit": deposit,
                "size": random.randint(5, 12),
                "walk": random.randint(1, 15),

                # 전월세 비율
                "jeonse_ratio": deposit / rent
            })

    return pd.DataFrame(rows)

rooms = make_rooms()

# =========================
# 경제 시나리오 (사이드바) - 🛠 변동 폭 대폭 확대
# =========================
scenario = st.sidebar.selectbox(
    "경제 시나리오 선택",
    ["보통", "경기호황", "경기침체"]
)

if scenario == "경기호황":
    # 고금리, 고환율, 고물가, 수요 폭발, 공급 부족 상황 극대화
    macro = {"interest": 6.5, "exchange": 1550, "inflation": 5.5, "demand": 1.6, "supply": 0.6}
elif scenario == "경기침체":
    # 초저금리, 저환율, 디플레이션 우려, 수요 급감, 공급 과잉 상황 극대화
    macro = {"interest": 1.5, "exchange": 1150, "inflation": 0.5, "demand": 0.5, "supply": 1.5}
else:
    macro = {"interest": 3.5, "exchange": 1350, "inflation": 2.5, "demand": 1.0, "supply": 1.0}

# =========================
# 🤖 머신러닝 학습 데이터 생성 - 🛠 거시경제 가중치 강화
# =========================
@st.cache_data
def make_ml_data(df):
    data = []

    for _, r in df.iterrows():
        for _ in range(5): # 데이터 다양성을 위해 반복 횟수 증가 (3 -> 5)

            interest = random.uniform(1.0, 7.0)
            exchange = random.uniform(1100, 1600)
            inflation = random.uniform(0.0, 6.0)
            demand = random.uniform(0.4, 1.8)
            supply = random.uniform(0.4, 1.8)

            # 🛠 거시경제 변수들이 월세에 미치는 영향(가중치)을 크게 늘렸습니다.
            rent = (
                r["rent"]
                + (interest - 3.5) * 12       # 금리 영향력 2배 (6 -> 12)
                + (exchange - 1300) * 0.05    # 환율 영향력 확대 (0.02 -> 0.05)
                + (inflation - 2.5) * 8       # 물가 영향력 2배 (4 -> 8)
                + (demand - 1.0) * 25         # 수요 영향력 대폭 확대 (12 -> 25)
                - (supply - 1.0) * 20         # 공급 영향력 대폭 확대 (12 -> 20)

                + (r["jeonse_ratio"] * 0.01)
                + (r["walk"] * -0.7)
                + (r["size"] * 0.5)

                + random.uniform(-2, 2)
            )

            data.append([
                r["size"],
                r["walk"],
                r["deposit"],
                r["jeonse_ratio"],
                interest,
                exchange,
                inflation,
                demand,
                supply,
                rent
            ])

    return pd.DataFrame(data, columns=[
        "size","walk","deposit","jeonse_ratio",
        "interest","exchange","inflation","demand","supply",
        "rent"
    ])

ml_data = make_ml_data(rooms)

# =========================
# 🤖 모델 학습
# =========================
@st.cache_resource
def train_model(data):
    X = data.drop("rent", axis=1)
    y = data["rent"]

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )
    model.fit(X, y)
    return model

model = train_model(ml_data)

# =========================
# 예측 함수
# =========================
def predict(row):
    X = pd.DataFrame([row])
    return model.predict(X)[0]

# =========================
# UI 배치 및 지도 출력
# =========================
col1, col2 = st.columns([1, 2])

with col1:
    selected = st.selectbox("🏫 대학 선택", list(UNIV.keys()))

    st.subheader(f"🌍 현재 경제 상태 ({scenario})")
    
    macro_ko = {
        "지표": ["기준금리", "환율", "물가상승률", "수요 지수", "공급 지수"],
        "수치": [
            f"{macro['interest']}%",
            f"{macro['exchange']}원",
            f"{macro['inflation']}%",
            f"{macro['demand']}x",
            f"{macro['supply']}x"
        ]
    }
    df_macro = pd.DataFrame(macro_ko)
    st.dataframe(df_macro, hide_index=True, use_container_width=True)

filtered = rooms[rooms["univ"] == selected]
lat, lon = UNIV[selected]

with col2:
    m = folium.Map(location=[lat, lon], zoom_start=15)

    folium.Marker([lat, lon], tooltip=selected,
                 icon=folium.Icon(color="red")).add_to(m)

    for _, r in filtered.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=5,
            color="blue",
            fill=True,
            tooltip=f"{r['name']} | {r['rent']}만원"
        ).add_to(m)

    map_data = st_folium(m, height=550)

clicked = None

if map_data and map_data.get("last_object_clicked"):
    c = map_data["last_object_clicked"]
    temp = filtered.copy()
    temp["d"] = (temp["lat"]-c["lat"])**2 + (temp["lon"]-c["lng"])**2
    clicked = temp.sort_values("d").iloc[0]

# =========================
# 분석 결과 출력 영역
# =========================
st.markdown("---")

if clicked is not None:

    feature = {
        "size": clicked["size"],
        "walk": clicked["walk"],
        "deposit": clicked["deposit"],
        "jeonse_ratio": clicked["jeonse_ratio"],

        "interest": macro["interest"],
        "exchange": macro["exchange"],
        "inflation": macro["inflation"],
        "demand": macro["demand"],
        "supply": macro["supply"]
    }

    pred = predict(feature)

    avg = filtered["rent"].mean()
    diff = clicked["rent"] - avg

    score = max(50, min(100, int(100 - abs(diff)*1.5)))

    c1, c2, c3 = st.columns(3)

    c1.metric("현재 매물 월세", f"{clicked['rent']}만원")
    c2.metric("ML 예측 월세", f"{pred:.1f}만원")
    c3.metric("AI 점수", f"{score}점")

    st.subheader("📊 핵심 분석")

    st.write(f"""
- 전세/월세 비율: {clicked['jeonse_ratio']:.2f}
- 도보 거리: {clicked['walk']}분
- 면적: {clicked['size']}평
- 보증금: {clicked['deposit']}만원
""")

    st.subheader("📈 미래 예측 (6개월 변동 시뮬레이션)")

    # 머신러닝 모델을 사용한 시나리오 기반 미래 예측
    future_predictions = []
    
    for i in range(1, 7):
        future_feature = feature.copy()
        
        # 🛠 미래 시나리오 변화폭도 체감되도록 조금 더 뚜렷하게 조정
        if scenario == "경기호황":
            future_feature["interest"] += i * 0.15     # 호황기에는 금리가 계속 추가 인상되는 시나리오
            future_feature["inflation"] += i * 0.1
        elif scenario == "경기침체":
            future_feature["interest"] -= i * 0.1      # 침체기에는 금리를 더 인하하는 시나리오
            future_feature["inflation"] -= i * 0.05
        else:
            future_feature["interest"] += i * 0.05
            future_feature["inflation"] += i * 0.02
        
        future_pred = predict(future_feature)
        future_predictions.append(future_pred)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(1, 7), future_predictions, marker="o", color="#e34a33" if scenario=="경기호황" else "#3182bd", linewidth=2)
    ax.set_xlabel("개월 뒤")
    ax.set_ylabel("예측 월세 (만원)")
    ax.set_xticks(range(1, 7))
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)

    st.subheader("🤖 AI 설명")

    st.info(f"""
머신러닝 기반 분석:

✔ 전세/월세 구조 반영  
✔ 금리/환율/물가 반영  
✔ 수요/공급 반영  
✔ 지역 접근성 반영  

선택하신 **[{scenario}]** 시나리오 요소를 가중 반영한 모델의 최종 결과입니다.
예측 월세: **{pred:.1f}만원**
""")

else:
    st.info("지도에서 매물을 클릭하면 해당 시나리오를 바탕으로 ML 분석이 시작됩니다.")
