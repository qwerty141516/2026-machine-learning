import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import folium

from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# =========================
# 🌐 CONFIG
# =========================
st.set_page_config(page_title="Macro Rent AI", layout="wide")

st.markdown("""
<h1 style='text-align:center;'>🏠 Macro Rent AI Simulator</h1>
<p style='text-align:center;color:gray;'>금리 · 환율 · 경제 변화 기반 월세 예측</p>
""", unsafe_allow_html=True)

# =========================
# 🏦 MACRO SLIDERS
# =========================
st.sidebar.header("📊 경제 변수")

interest = st.sidebar.slider("금리 (%)", 1.0, 6.0, 3.5, 0.1)
inflation = st.sidebar.slider("물가 상승률 (%)", 0.0, 6.0, 2.5, 0.1)
exchange = st.sidebar.slider("환율 (USD/KRW)", 1100, 1600, 1350, 10)

event = st.sidebar.slider("지역 경제 이벤트 (-1 ~ 1)", -1.0, 1.0, 0.0, 0.1)
supply = st.sidebar.slider("공급 지수 (낮을수록 부족)", 0.5, 1.5, 1.0, 0.1)

macro = {
    "interest": interest,
    "inflation": inflation,
    "exchange": exchange,
    "event": event,
    "supply": supply
}

# =========================
# 🏠 SAMPLE RENT DATA
# =========================
def base_rent():
    return 70

def predict_rent(m):
    rent = (
        base_rent()
        + m["interest"] * 4
        + (m["exchange"] - 1300) * 0.03
        + m["inflation"] * 3
        + m["event"] * 10
        - m["supply"] * 12
    )
    return max(30, rent)

current_rent = predict_rent(macro)

# =========================
# 📈 PRICE CHANGE ANIMATION
# =========================
st.markdown("## 📊 가격 변화 시뮬레이션")

history = []
base = 65

for i in range(10):
    noise = random.uniform(-2, 2)
    value = base + i * (current_rent - base) / 9 + noise
    history.append(value)

chart_area = st.empty()

for i in range(1, len(history)+1):
    fig, ax = plt.subplots()
    ax.plot(range(i), history[:i], marker="o")
    ax.set_title("월세 변화 추이")
    ax.set_ylim(min(history)-5, max(history)+5)
    chart_area.pyplot(fig)
    time.sleep(0.15)

# =========================
# 📊 RESULT DASHBOARD
# =========================
st.markdown("## 💰 현재 분석 결과")

col1, col2 = st.columns(2)

with col1:
    st.metric("예측 월세", f"{current_rent:.1f} 만원")

with col2:
    st.metric("기준 대비 변화", f"{((current_rent/base_rent())-1)*100:.1f}%")

# =========================
# 🤖 GPT STYLE EXPLANATION
# =========================
st.markdown("## 🤖 AI 분석 리포트")

reason = []

if interest > 4:
    reason.append("금리 상승으로 인해 대출 부담이 증가하여 월세 상승 압력이 발생합니다.")

if inflation > 3:
    reason.append("물가 상승으로 인해 전반적인 임대료가 상승하는 구조입니다.")

if exchange > 1450:
    reason.append("환율 상승으로 외국인 수요 변화 및 투자 자금 유입 가능성이 증가합니다.")

if event > 0.3:
    reason.append("지역 개발 및 호재로 인해 수요가 증가하고 있습니다.")

elif event < -0.3:
    reason.append("지역 악재로 인해 수요 감소 압력이 존재합니다.")

if supply < 0.8:
    reason.append("공급 부족으로 인해 가격 상승 압력이 강합니다.")

elif supply > 1.2:
    reason.append("공급 과잉으로 인해 가격 하락 요인이 존재합니다.")

st.markdown(f"""
### 🧠 AI 분석 요약

현재 경제 조건에서 월세는 **{current_rent:.1f}만원** 수준으로 예측됩니다.

**핵심 요인:**
""")

for r in reason:
    st.write("• " + r)

# =========================
# 📊 MINI CHART (ECONOMY)
# =========================
st.markdown("## 📉 경제 변수 영향도")

labels = ["금리", "물가", "환율", "이벤트", "공급"]
values = [
    interest * 4,
    inflation * 3,
    (exchange - 1300) * 0.03,
    event * 10,
    -supply * 12
]

fig, ax = plt.subplots()
ax.bar(labels, values)
ax.set_title("경제 요인별 월세 영향")
st.pyplot(fig)
