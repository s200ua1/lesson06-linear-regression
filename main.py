
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고, "
    "선형 회귀를 이용해 선택한 연도의 예상 기온을 확인합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
기준연도 = 2025
최소관측일 = 300


# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    필요한열 = ["날짜", "평균기온"]
    for col in 필요한열:
        if col not in df.columns:
            raise ValueError(f"데이터에 '{col}' 열이 없습니다.")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    return df


try:
    data = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# -----------------------------
# 연도별 평균기온 계산
# 2025년까지, 관측일 300일 이상인 해만 사용
# -----------------------------
연도별 = (
    data[data["연도"] <= 기준연도]
    .groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

연도별 = 연도별[
    연도별["관측일수"] >= 최소관측일
].copy()

연도별 = 연도별.sort_values("연도").reset_index(drop=True)

if len(연도별) < 2:
    st.error(
        "회귀 직선을 만들 데이터가 부족합니다. "
        "날짜와 평균기온 열을 확인해 주세요."
    )
    st.stop()


# -----------------------------
# 선형 회귀 계산
# -----------------------------
x = 연도별["연도"].to_numpy(dtype=float)
y = 연도별["연평균기온"].to_numpy(dtype=float)

기울기, 절편 = np.polyfit(x, y, 1)
상관계수 = np.corrcoef(x, y)[0, 1]

시작연도 = int(연도별["연도"].min())
끝연도 = int(연도별["연도"].max())
사용연도수 = len(연도별)


def 예상기온(연도):
    return 기울기 * 연도 + 절편


# -----------------------------
# 학습 데이터 정보
# -----------------------------
st.subheader("📊 회귀 분석에 사용한 데이터")

c1, c2, c3, c4 = st.columns(4)

c1.metric("사용한 연도 수", f"{사용연도수:,}년")
c2.metric("시작 연도", f"{시작연도}년")
c3.metric("끝 연도", f"{끝연도}년")
c4.metric("상관계수", f"{상관계수:.3f}")

st.caption(
    f"기준: {기준연도}년까지의 자료 중 "
    f"관측일수가 {최소관측일}일 이상인 연도"
)


# -----------------------------
# 연도 슬라이더와 예상 기온
# -----------------------------
st.subheader("🌡️ 원하는 연도의 예상 기온")

선택연도 = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

선택기온 = 예상기온(선택연도)

m1, m2 = st.columns(2)
m1.metric(
    f"{선택연도}년 예상 연평균기온",
    f"{선택기온:.2f} °C"
)
m2.metric(
    "회귀 직선의 기울기",
    f"{기울기:.4f} °C/년"
)

if 선택연도 > 끝연도:
    st.info(
        "선택한 연도는 학습 데이터의 마지막 연도 이후입니다. "
        "과거 추세를 연장한 추정치이므로 실제 기온과 다를 수 있습니다."
    )
elif 선택연도 < 시작연도:
    st.info(
        "선택한 연도는 학습 데이터의 시작 연도보다 이전입니다. "
        "회귀 직선을 과거로 연장한 추정치입니다."
    )


# -----------------------------
# 산점도와 회귀 직선
# -----------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

직선연도 = np.linspace(1900, 2100, 300)
직선기온 = 예상기온(직선연도)

fig = go.Figure()

# 실제 관측 자료
fig.add_trace(
    go.Scatter(
        x=연도별["연도"],
        y=연도별["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=연도별[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=직선연도,
        y=직선기온,
        mode="lines",
        name="선형 회귀 직선",
        line=dict(width=3)
    )
)

# 선택한 연도의 예상 기온
fig.add_trace(
    go.Scatter(
        x=[선택연도],
        y=[선택기온],
        mode="markers",
        name=f"{선택연도}년 예상 기온",
        marker=dict(size=14, symbol="star"),
        hovertemplate=(
            "선택 연도: %{x}년<br>"
            "예상 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend_title="범례",
    height=580
)

fig.update_xaxes(range=[1900, 2100], dtick=20)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 해석 안내
# -----------------------------
st.subheader("💡 그래프 읽는 방법")

st.write(
    f"- 산점도는 관측 자료로 계산한 연평균기온을 나타냅니다.\n"
    f"- 회귀 직선은 {시작연도}~{끝연도}년의 자료에 "
    "가장 잘 맞도록 계산한 직선입니다.\n"
    f"- 상관계수는 {상관계수:.3f}입니다. "
    "양수이면 연도와 기온이 함께 증가하는 경향, "
    "음수이면 반대 경향을 나타냅니다.\n"
    f"- 직선의 기울기는 연평균기온의 선형 추세가 "
    f"연간 약 {기울기:.4f} °C 변한다는 뜻입니다."
)

st.warning(
    "주의: 이 앱은 과거 자료에 직선을 맞춘 단순 선형 회귀 모델입니다. "
    "온실가스 배출, 자연 변동, 도시 환경 변화 등은 따로 반영하지 않습니다. "
    "특히 먼 미래의 값은 실제 기후 예측이 아닌 추세 연장값으로 해석하세요."
)

with st.expander("사용한 연도별 데이터 보기"):
    st.dataframe(
        연도별.rename(
            columns={
                "연도": "연도",
                "연평균기온": "연평균기온(°C)",
                "관측일수": "관측일수(일)"
            }
        ),
        use_container_width=True,
        hide_index=True
    )
