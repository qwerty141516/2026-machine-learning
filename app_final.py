import streamlit as st
import pandas as pd
import random
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

st.set_page_config(page_title="서울 대학가 원룸 월세 예측 모델", layout="wide")

st.markdown("""
<h1 style='text-align:center;'>🏠 서울 대학가 원룸 월세 예측 모델</h1>
<p style='text-align:center;color:gray;'>AI 기반 거시경제·지역 수요 분석 시스템</p>
""", unsafe_allow_html=True)

UNIV = {
    "연세대": (37.5658, 126.9386),
    "서울대": (37.4599, 126.9519),
    "고려대": (37.5894, 127.0324)
}

@st.cache_data
def make_rooms():
    rows = []
    random.seed(42)
    for u, (lat, lon) in UNIV.items():
        for i in range(30):
            rows.append({
                "name": f"{u} 원룸 {i+1}",
                "univ": u,
                "lat": lat + random.uniform(-0.004, 0.004),
                "lon": lon + random.uniform(-0.004, 0.004),
                "rent": random.randint(40, 90),
                "deposit": random.randint(500, 3000),
                "size": random.randint(5, 12),
                "walk": random.randint(1, 15)
            })
    return pd.DataFrame(rows)

rooms = make_rooms()

scenario = st.sidebar.selectbox(
    "경제 시나리오",
    ["보통", "경기호황", "경기침체"]
)

if scenario == "경기호황":
    macro = {"interest":5.0,"exchange":1450,"inflation":4.0,"demand":1.2,"supply":0.9}
elif scenario == "경기침체":
    macro = {"interest":2.5,"exchange":1250,"inflation":1.5,"demand":0.8,"supply":1.2}
else:
    macro = {"interest":3.5,"exchange":1350,"inflation":2.5,"demand":1.0,"supply":1.0}

def predict_rent(base):
    return max(
        20,
        base
        + (macro["interest"] - 3) * 6
        + (macro["exchange"] - 1300) * 0.02
        + (macro["inflation"] - 2) * 5
        + (macro["demand"] - 1) * 15
        - (macro["supply"] - 1) * 15
    )

col1, col2 = st.columns([1, 2])

with col1:
    selected = st.selectbox("🏫 대학 선택", list(UNIV.keys()))
    st.subheader("🌍 경제 상태")
    st.write(f"금리: {macro['interest']}%")
    st.write(f"환율: {macro['exchange']}원")
    st.write(f"물가: {macro['inflation']}%")
    st.write(f"수요지수: {macro['demand']}")
    st.write(f"공급지수: {macro['supply']}")

    with st.expander("경제지표 쉽게 이해하기"):
        st.write("금리↑ → 월세 상승 가능")
        st.write("수요↑ → 월세 상승 가능")
        st.write("공급↑ → 월세 하락 가능")

filtered = rooms[rooms["univ"] == selected]
lat, lon = UNIV[selected]

with col2:
    m = folium.Map(location=[lat, lon], zoom_start=15)

    folium.Marker(
        [lat, lon],
        tooltip=selected,
        icon=folium.Icon(color="red")
    ).add_to(m)

    for _, r in filtered.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=5,
            color="blue",
            fill=True,
            tooltip=f"{r['name']} | {r['rent']}만원"
        ).add_to(m)

    map_data = st_folium(m, height=550, width=None)

clicked = None

if map_data and map_data.get("last_object_clicked"):
    c = map_data["last_object_clicked"]
    temp = filtered.copy()
    temp["d"] = (temp["lat"]-c["lat"])**2 + (temp["lon"]-c["lng"])**2
    clicked = temp.sort_values("d").iloc[0]

st.markdown("---")

if clicked is not None:

    base = clicked["rent"]
    pred = predict_rent(base)

    avg_rent = filtered["rent"].mean()
    diff = base - avg_rent

    score = max(50, min(100, int(100 - abs(diff)*1.5)))

    if pred < 55:
        grade = "🟢 저렴"
    elif pred < 75:
        grade = "🟡 보통"
    else:
        grade = "🔴 비쌈"

    c1, c2, c3 = st.columns(3)

    c1.metric("현재 월세", f"{base}만원")
    c2.metric("예측 월세", f"{pred:.1f}만원")
    c3.metric("AI 추천도", f"{score}점")

    st.subheader("📋 매물 정보")

    st.write(f"보증금: {clicked['deposit']}만원")
    st.write(f"면적: {clicked['size']}평")
    st.write(f"학교까지 도보: {clicked['walk']}분")
    st.write(f"월세 등급: {grade}")

    st.subheader("📊 주변 시세 비교")

    if diff > 0:
        st.warning(f"주변 평균보다 약 {diff:.1f}만원 비쌉니다.")
    else:
        st.success(f"주변 평균보다 약 {abs(diff):.1f}만원 저렴합니다.")

    st.subheader("📈 월세 변화 예측")

    future = [pred + random.uniform(-3, 3) + i for i in range(6)]

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(range(1,7), future, marker="o")
    ax.set_xlabel("개월")
    ax.set_ylabel("월세(만원)")
    st.pyplot(fig)

    st.subheader("🤖 AI 한눈에 요약")

    st.info(
        f"""
현재 월세는 {base}만원이며 예측 월세는 {pred:.1f}만원입니다.

예상 변동 이유
✔ 지역 수요 반영
✔ 경제 상황 반영
✔ 공급 수준 반영

AI 추천도는 {score}점으로 분석되었습니다.
"""
    )

else:
    st.info("지도에서 파란 매물을 클릭하면 분석이 시작됩니다.")
