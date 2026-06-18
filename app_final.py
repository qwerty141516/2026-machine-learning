import streamlit as st
import pandas as pd
import random
import math
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
<p style='text-align:center;color:gray;'>AI 기반 거시경제 분석 & 원룸 스마트 가이드</p>
""", unsafe_allow_html=True)

# =========================
# 대학 좌표
# =========================
UNIV = {
    "연세대": (37.5658, 126.9386),
    "서울대": (37.4599, 126.9519),
    "고려대": (37.5894, 127.0324)
}

# 실제 대학가 주거지역 좌표
HOUSING_AREAS = {
    "연세대": [(37.5595,126.9360),(37.5620,126.9345),(37.5670,126.9355),(37.5680,126.9390),(37.5615,126.9410)],
    "서울대": [(37.4790,126.9520),(37.4815,126.9540),(37.4830,126.9500),(37.4770,126.9560),(37.4850,126.9525)],
    "고려대": [(37.5860,127.0300),(37.5885,127.0315),(37.5900,127.0295),(37.5910,127.0340),(37.5850,127.0345)]
}

def calc_walk_minutes(room_lat, room_lon, school_lat, school_lon):
    distance_km = (((room_lat - school_lat)**2 + (room_lon - school_lon)**2) ** 0.5) * 111
    return max(1, int(distance_km * 12))


# =========================
# 가상 매물 생성
# =========================
@st.cache_data
def make_rooms():
    rows = []
    random.seed(42)

    for u, (school_lat, school_lon) in UNIV.items():
        for i in range(80):

            deposit = random.randint(500, 3000)
            rent = random.randint(40, 90)

            center_lat, center_lon = random.choice(HOUSING_AREAS[u])

            distance = random.uniform(0.002, 0.007)
            angle = random.uniform(0, 2 * math.pi)

            lat = center_lat + distance * math.cos(angle)
            lon = center_lon + distance * math.sin(angle)

            walk = calc_walk_minutes(
                lat, lon,
                school_lat, school_lon
            )

            rows.append({
                "name": f"{u} 원룸 {i+1}",
                "univ": u,
                "lat": lat,
                "lon": lon,
                "rent": rent,
                "deposit": deposit,
                "size": random.randint(5, 12),
                "walk": walk,
                "jeonse_ratio": deposit / rent
            })

    return pd.DataFrame(rows)

rooms = make_rooms()

# =========================
# 경제 시나리오 (사이드바)
# =========================
scenario = st.sidebar.selectbox(
    "경제 시나리오 선택",
    ["보통", "경기호황", "경기침체"]
)

if scenario == "경기호황":
    macro = {"interest": 6.5, "exchange": 1350, "inflation": 5.5, "demand": 1.9, "supply": 0.4}
elif scenario == "경기침체":
    macro = {"interest": 1.5, "exchange": 1650, "inflation": 0.5, "demand": 0.3, "supply": 1.8}
else:
    macro = {"interest": 3.5, "exchange": 1515, "inflation": 2.5, "demand": 1.0, "supply": 1.0}

# =========================
# 🤖 머신러닝 학습 데이터 생성
# =========================
@st.cache_data
def make_ml_data(df):
    data = []

    for _, r in df.iterrows():
        for _ in range(5): 

            interest = random.uniform(1.0, 7.0)
            exchange = random.uniform(1200, 1800)
            inflation = random.uniform(0.0, 6.0)
            demand = random.uniform(0.2, 2.2)
            supply = random.uniform(0.2, 2.2)

            rent = (
                r["rent"]
                + (interest - 3.5) * 6       
                - (exchange - 1515) * 0.05    
                + (inflation - 2.5) * 5       
                + (demand - 1.0) * 25         
                - (supply - 1.0) * 20         

                + (r["jeonse_ratio"] * 0.01)
                + (r["walk"] * -0.5)
                + (r["size"] * 0.5)

                + random.uniform(-6, 6) 
            )

            if rent < 30:
                rent = random.uniform(30, 38)
            elif rent > 125:
                rent = random.uniform(115, 125)

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
    m = folium.Map(location=[lat, lon], zoom_start=14)

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

    # 🛠 [AI 점수 수식 변경: 0점 ~ 100점 스케일링]
    price_gap = clicked["rent"] - pred
    
    # 기본 균형 점수를 80점으로 두고, 오차값에 따른 민감도를 높여 0~100 범위를 넓게 쓰도록 수정
    raw_score = 80 - (price_gap * 3.5)
    
    # 입지 가감산 적용
    if clicked['walk'] <= 5:
        raw_score += 5
    elif clicked['walk'] >= 12:
        raw_score -= 5
        
    # 최종 점수를 0점부터 100점 사이로 강제 제한 (Clip)
    score = max(0, min(100, int(raw_score)))

    c1, c2, c3 = st.columns(3)

    c1.metric("현재 매물 원래 월세", f"{clicked['rent']}만원")
    c2.metric("ML 예측 적정 월세", f"{pred:.1f}만원")
    c3.metric("AI 추천 점수", f"{score}점")

    # 버튼식(Expander) 추천 점수 산출 근거 확인 영역
    with st.expander("🔍 AI 추천 점수 산출 근거 확인하기"):
        st.markdown(f"### 📋 점수 세부 진단 내역 (최종 점수: **{score}점** / 100점)")
        st.write(f"- **기본 가성비 베이스 점수**: `80점`")
        
        if price_gap > 0:
            st.write(f"- ❌ **가격 고평가 감점**: 현재 매물가({clicked['rent']}만원)가 ML 예측 적정가({pred:.1f}만원)보다 **{price_gap:.1f}만원** 더 높습니다. 시장 거품이 확인되어 **{price_gap*3.5:.1f}점 감점**되었습니다.")
        else:
            st.write(f"-  **가격 가성비 가점**: 현재 매물가({clicked['rent']}만원)가 ML 예측 적정가({pred:.1f}만원)보다 **{abs(price_gap):.1f}만원** 더 저렴합니다. 탁월한 가성비 혜택으로 **{abs(price_gap)*3.5:.1f}점 가점**되었습니다.")
            
        if clicked['walk'] <= 5:
            st.write("- ⭐ **초역세권/대학 인접 가점**: 학교까지의 도보 거리 가산점 `+5점`이 추가되었습니다.")
        elif clicked['walk'] >= 12:
            st.write("- 🏃 **입지 원거리 감점**: 학교까지 도보 이동 거리 부담 요인으로 인해 `-5점`이 차감되었습니다.")
            
        st.caption("※ 본 점수는 0점(최하 추천)부터 100점(최고 추천) 사이의 스케일로 환산된 지표입니다.")

    tab1, tab2, tab3 = st.tabs(["📊 ML 분석 & 예측", "📝 매물 방문 체크리스트", "💡 원룸 계약 필수 상식"])

    with tab1:
        col_info1, col_info2 = st.columns([1, 1])
        
        with col_info1:
            st.subheader("📊 매물 특성")
            st.write(f"""
            - **전세/월세 비율**: {clicked['jeonse_ratio']:.2f}
            - **도보 거리**: 학교까지 {clicked['walk']}분
            - **방 크기**: {clicked['size']}평
            - **보증금**: {clicked['deposit']}만원
            """)
        
        with col_info2:
            st.subheader("🤖 AI 매물 진단 보고서")
            
            # 0~100 대역에 맞게 멘트 조건문 세분화
            if score >= 85:
                price_eval = "💡 이 매물은 현재 시나리오 예측가 대비 월세 효율이 대단히 좋아 **가성비가 최고 수준인 매물**입니다."
            elif score <= 40:
                price_eval = "⚠️ 이 매물은 거시경제 지표 및 예측가 대비 임대료 거품이 대단히 심한 **비추천(고평가) 매물**입니다."
            else:
                price_eval = "⚖️ 이 매물은 시장 트렌드와 모델 예측치 범위 안에 안정적으로 들어오는 **평균적이고 무난한 가격의 매물**입니다."
                
            if scenario == "경기호황":
                macro_eval = f"현재 **{scenario}** 상황으로 인해 대학가 유입 수요가 폭발하고 공급이 부족한 상태입니다. 모델은 이 매물의 적정 월세를 호황기 프리미엄이 붙은 **{pred:.1f}만원** 선으로 바라보고 있습니다."
            elif scenario == "경기침체":
                macro_eval = f"반면 현재 **{scenario}** 상황에서는 대학가 인구 수요가 급감하고 매물 공급이 과잉되어 시장이 위축되어 있습니다. 이에 따라 모델은 이 방의 적정 가치를 하락 안정세인 **{pred:.1f}만원** 선으로 진단했습니다."
            else:
                macro_eval = f"안정적인 **{scenario}** 경제 지표 아래에서 머신러닝 모델이 예측한 적정 월세는 **{pred:.1f}만원**입니다."

            st.write(f"📝 {price_eval} {macro_eval}")

        st.subheader("📈 미래 예측 (6개월 변동 시뮬레이션)")
        future_predictions = []
        for i in range(1, 7):
            future_feature = feature.copy()
            if scenario == "경기호황":
                future_feature["interest"] += i * 0.05     
                future_feature["inflation"] += i * 0.03
                future_feature["exchange"] -= i * 5       
            elif scenario == "경기침체":
                future_feature["interest"] -= i * 0.05      
                future_feature["inflation"] -= i * 0.02
                future_feature["exchange"] += i * 5       
            else:
                future_feature["interest"] += i * 0.02
                future_feature["inflation"] += i * 0.01

            future_pred = predict(future_feature)
            future_pred = max(30.0, min(125.0, future_pred))
            future_predictions.append(future_pred)

        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.plot(range(1, 7), future_predictions, marker="o", color="#e34a33" if scenario=="경기호황" else "#3182bd", linewidth=2)
        ax.set_xlabel("개월 뒤")
        ax.set_ylabel("예측 월세 (만원)")
        ax.set_xticks(range(1, 7))
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    with tab2:
        st.subheader(f"🔍 {clicked['name']} 현장 방문 체크리스트")
        st.caption("방을 직접 보러 갔을 때 아래 항목들을 체크해보세요!")
        
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.checkbox("🚿 수압 & 배수 (화장실 변기 물 내리면서 세면대 켜보기)")
            st.checkbox("☀️ 채광 및 환기 (창문 방향 확인 및 곰팡이 흔적 체크)")
            st.checkbox("🚪 방음 상태 (복도 소음이나 옆방 말소리가 크게 들리는지)")
        with c_col2:
            st.checkbox("🔌 옵션 상태 (에어컨, 세탁기, 냉장고 정상 작동 여부)")
            st.checkbox("🔒 보안 인프라 (공동현관 도어락, 주변 CCTV 위치 확인)")
            st.checkbox("🔒 수압/보일러 (온수가 끊김 없이 잘 나오는지)")

    with tab3:
        st.subheader("💡 방 구하기 전 필수 체크 법률/금융 가이드")
        
        st.info("""
        **1. 등기부등본 확인 (근저당 체크)**
        - 계약 전 '인터넷등기소'에서 등기부등본을 열람해 **[을구]의 근저당권(융자)**을 확인하세요.
        - **[건물 매매가]의 60~70% 이상**이 빚(근저당+선순위 보증금)으로 묶여있다면 조심해야 합니다!
        """)
        
        st.success("""
        **2. 계약 당일 필수 액션 (대항력 갖추기)**
        - 이사(인도) 후 잔금을 치른 날, 곧바로 주민센터나 인터넷 법원등기소에서 **[전입신고]와 [확정일자]**를 받으세요.
        - 보증금을 법적으로 안전하게 보호받기 위한 최우선 방어벽입니다.
        """)
        
        st.warning("""
        **3. 중개수수료(복비) 바가지 예방**
        - 중개수수료는 법정 상한 요율이 정해져 있습니다 (보통 대학가 원룸 요율은 거래금액의 0.4%~0.5% 내외).
        - 계약서 쓰기 전 미리 요율을 확인하고, 현금영수증을 꼭 요청하세요.
        """)

else:
    st.info("지도에서 매물을 클릭하면 해당 시나리오를 바탕으로 ML 분석 및 가이드가 시작됩니다.")
