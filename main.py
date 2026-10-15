
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =====================================
# 1. 기본 설정
# =====================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 과거 기온 데이터를 분석하여 연평균기온의 추세를 확인합니다.")

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

기준연도 = 2025
최소관측일 = 300


# =====================================
# 2. 데이터 불러오기
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    필요한열 = ["날짜", "평균기온"]
    없는열 = [col for col in 필요한열 if col not in df.columns]

    if 없는열:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {없는열}")

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
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# =====================================
# 3. 연도별 평균기온 계산
# =====================================
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
    st.error("회귀 분석에 필요한 데이터가 부족합니다.")
    st.stop()


# =====================================
# 4. 전체 기간 회귀 분석
# =====================================
x = 연도별["연도"].to_numpy(dtype=float)
y = 연도별["연평균기온"].to_numpy(dtype=float)

기울기, 절편 = np.polyfit(x, y, 1)
상관계수 = np.corrcoef(x, y)[0, 1]

시작연도 = int(연도별["연도"].min())
끝연도 = int(연도별["연도"].max())
사용연도수 = len(연도별)

# 1년당 변화량을 100년당 변화량으로 환산
전체100년변화 = 기울기 * 100


def 예상기온(연도):
    return 기울기 * 연도 + 절편


# =====================================
# 5. 최근 20년 회귀 분석
# =====================================
# 마지막 유효 관측 연도를 기준으로 20년 범위 설정
최근20시작 = 끝연도 - 19

최근20 = 연도별[
    (연도별["연도"] >= 최근20시작)
    & (연도별["연도"] <= 끝연도)
].copy()

if len(최근20) >= 2:
    최근기울기, 최근절편 = np.polyfit(
        최근20["연도"].to_numpy(dtype=float),
        최근20["연평균기온"].to_numpy(dtype=float),
        1
    )
    최근100년변화 = 최근기울기 * 100
else:
    최근기울기 = np.nan
    최근절편 = np.nan
    최근100년변화 = np.nan


# =====================================
# 6. 분석에 사용한 데이터 정보
# =====================================
st.subheader("📊 분석 데이터")

c1, c2, c3 = st.columns(3)

c1.metric("사용한 연도 수", f"{사용연도수}년")
c2.metric("시작 연도", f"{시작연도}년")
c3.metric("끝 연도", f"{끝연도}년")

st.caption(
    f"{기준연도}년까지의 자료 중 연평균기온을 계산할 수 있고 "
    f"관측일이 {최소관측일}일 이상인 연도만 사용했습니다."
)


# =====================================
# 7. 100년당 기온 변화 비교
# =====================================
st.subheader("🌡️ 100년당 기온 변화 비교")

c1, c2 = st.columns(2)

with c1:
    st.metric(
        label="전체 기간의 기온 변화",
        value=f"{전체100년변화:+.2f} °C"
    )
    st.caption(
        f"전체 분석 기간: {시작연도}~{끝연도}년"
    )

with c2:
    if np.isfinite(최근100년변화):
        st.metric(
            label="최근 20년 추세의 100년당 환산값",
            value=f"{최근100년변화:+.2f} °C"
        )
        st.caption(
            f"분석 기간: {int(최근20['연도'].min())}"
            f"~{int(최근20['연도'].max())}년 "
            f"({len(최근20)}개 유효 연도)"
        )
    else:
        st.warning("최근 20년의 회귀 분석에 필요한 자료가 부족합니다.")

st.info(
    "양수는 상승 추세, 음수는 하강 추세를 뜻합니다. "
    "최근 20년의 값을 100년 기준으로 환산한 것이므로 "
    "실제로 100년 동안 관측한 결과는 아닙니다."
)


# =====================================
# 8. 연도 선택 및 예상 기온
# =====================================
st.subheader("🔮 예상 연평균기온")

선택연도 = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

선택기온 = 예상기온(선택연도)

st.metric(
    label=f"{선택연도}년 예상 연평균기온",
    value=f"{선택기온:.2f} °C"
)

if 선택연도 > 끝연도:
    st.warning(
        "관측 자료의 마지막 연도 이후를 예측한 값입니다. "
        "과거의 직선 추세를 연장한 추정치이며 실제 기온과 다를 수 있습니다."
    )
elif 선택연도 < 시작연도:
    st.warning(
        "관측 자료의 시작 연도 이전을 추정한 값입니다. "
        "회귀 직선을 과거로 연장한 결과입니다."
    )


# =====================================
# 9. 산점도 및 회귀 직선
# =====================================
st.subheader("📈 연도별 평균기온과 회귀 직선")

직선연도 = np.linspace(1900, 2100, 300)
직선기온 = 예상기온(직선연도)

fig = go.Figure()

# 실제 연평균기온 산점도
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

# 전체 기간 회귀 직선
fig.add_trace(
    go.Scatter(
        x=직선연도,
        y=직선기온,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3)
    )
)

# 선택한 연도의 예상값
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
    height=550,
    hovermode="closest",
    legend_title="범례"
)

fig.update_xaxes(range=[1900, 2100], dtick=20)

st.plotly_chart(fig, use_container_width=True)


# =====================================
# 10. 상관계수 및 해석
# =====================================
st.subheader("📖 분석 결과 해석")

st.metric(
    "연도와 연평균기온의 상관계수",
    f"{상관계수:.3f}"
)

st.write(
    f"- 회귀 분석에는 총 {사용연도수}개 연도의 자료를 사용했습니다.\n"
    f"- 분석 기간은 {시작연도}년부터 {끝연도}년까지입니다.\n"
    f"- 전체 기간의 선형 추세를 100년 기준으로 환산하면 "
    f"{전체100년변화:+.2f} °C입니다."
)

if np.isfinite(최근100년변화):
    차이 = 최근100년변화 - 전체100년변화

    st.write(
        f"- 최근 20년 추세의 100년당 환산값은 "
        f"{최근100년변화:+.2f} °C입니다.\n"
        f"- 최근 추세와 전체 기간 추세의 차이는 "
        f"{차이:+.2f} °C입니다."
    )

st.warning(
    "이 앱은 연도와 기온 사이의 선형 관계를 이용한 간단한 통계 모델입니다. "
    "온실가스 배출, 자연적인 기후 변동, 도시 환경 변화 등을 반영하지 않으므로 "
    "특히 먼 미래의 값은 실제 기후 예측으로 해석하면 안 됩니다."
)


# =====================================
# 11. 연도별 데이터 표
# =====================================
with st.expander("연도별 분석 데이터 확인하기"):
    표시용 = 연도별.rename(
        columns={
            "연평균기온": "연평균기온(°C)",
            "관측일수": "관측일수(일)"
        }
    )

    st.dataframe(
        표시용,
        use_container_width=True,
        hide_index=True
    )
