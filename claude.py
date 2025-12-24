import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
import ssl
import requests
import warnings
import pytz  # 한국 시간을 위해 라이브러리 추가

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# [1] 기본 설정 & SSL 강력 우회
# ---------------------------------------------------------
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 브라우저 탭 제목
st.set_page_config(page_title="WBD Arbitrage Dashboard", layout="wide")

# ---------------------------------------------------------
# [2] 세션 상태 초기화
# ---------------------------------------------------------
if 'wbd_vol' not in st.session_state: st.session_state['wbd_vol'] = 49.0
if 'nflx_vol' not in st.session_state: st.session_state['nflx_vol'] = 29.5
if 'correlation' not in st.session_state: st.session_state['correlation'] = None
if 'wbd_returns_data' not in st.session_state: st.session_state['wbd_returns_data'] = None
if 'nflx_returns_data' not in st.session_state: st.session_state['nflx_returns_data'] = None

# ---------------------------------------------------------
# [3] 날짜 설정 (사이드바에서 변경 가능하도록 수정)
# ---------------------------------------------------------
st.sidebar.header("📅 시뮬레이션 날짜 설정")
SIMULATED_TODAY = st.sidebar.date_input(
    "현재 시점 (Today)",
    value=datetime.date(2025, 12, 24),
    min_value=datetime.date(2025, 12, 1),
    max_value=datetime.date(2026, 1, 21)
)
TARGET_DATE = datetime.date(2026, 1, 21)

# ★ 투자 기간 및 연환산 계수 계산 (동적 변경)
INVEST_DAYS = (TARGET_DATE - SIMULATED_TODAY).days 
if INVEST_DAYS <= 0: INVEST_DAYS = 0 # 종료일 지나면 0 처리
ANNUAL_FACTOR = 365 / INVEST_DAYS if INVEST_DAYS > 0 else 0

# ---------------------------------------------------------
# [4] 함수 정의 (스마트 날짜 로직 & 에러 방지 유지)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def calculate_volatility_robust(ticker, start_date, end_date=None):
    if end_date is None: end_date = SIMULATED_TODAY
    
    # ★ 날짜 매핑 로직 (2025년 -> 2024년 데이터 연결)
    real_year = datetime.date.today().year 
    try:
        fetch_start = start_date.replace(year=real_year)
    except ValueError:
        fetch_start = start_date.replace(year=real_year, day=start_date.day-1)
        
    fetch_end = datetime.date.today() 

    if fetch_start > fetch_end:
        fetch_start = fetch_end - datetime.timedelta(days=1)

    def process_data(data):
        if not data.empty and len(data) > 1:
            col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            if isinstance(data.columns, pd.MultiIndex):
                try: prices = data.xs(col, axis=1, level=0)
                except: prices = data[col]
            else:
                prices = data[col]
            
            # [수정] 데이터프레임일 경우 Series로 강제 변환
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]

            prices = prices.dropna()
            
            mask = (prices.index.date >= fetch_start) & (prices.index.date <= fetch_end)
            prices = prices.loc[mask]
            
            daily_returns = prices.pct_change().dropna()
            
            if len(daily_returns) > 1:
                # 엑셀 STDEV.S (ddof=1) * 15.87 적용
                std_val = daily_returns.std(ddof=1)
                
                # [수정 핵심] 결과가 Series나 DataFrame이면 숫자(float)로 강제 변환
                if isinstance(std_val, (pd.Series, pd.DataFrame)):
                    if not std_val.empty:
                        std_val = std_val.iloc[0]
                    else:
                        return None, None
                
                vol = float(std_val) * 15.87 * 100
                return vol, daily_returns
        return None, None

    try:
        data = yf.download(ticker, start=fetch_start, end=fetch_end + datetime.timedelta(days=1), progress=False, threads=False)
        vol, ret = process_data(data)
        if vol is not None: return vol, ret
    except: pass
    
    try:
        session = requests.Session()
        session.verify = False
        start_ts = int(pd.Timestamp(fetch_start).timestamp())
        end_ts = int(pd.Timestamp(fetch_end + datetime.timedelta(days=1)).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_ts}&period2={end_ts}&interval=1d"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = session.get(url, headers=headers, timeout=5).json()
        result = resp['chart']['result'][0]
        timestamps = result['timestamp']
        closes = result['indicators']['quote'][0]['close']
        
        valid_data = []
        for t, c in zip(timestamps, closes):
            if c is not None:
                d_date = pd.to_datetime(t, unit='s').date()
                if fetch_start <= d_date <= fetch_end:
                    valid_data.append((t, c))
        
        if len(valid_data) > 2:
            ts, cs = zip(*valid_data)
            df = pd.DataFrame({'Close': cs}, index=pd.to_datetime(ts, unit='s'))
            daily_returns = df['Close'].pct_change().dropna()
            if len(daily_returns) > 1:
                std_val = daily_returns.std(ddof=1)
                # [수정 핵심] 여기도 안전장치 추가
                if isinstance(std_val, (pd.Series, pd.DataFrame)):
                    std_val = std_val.iloc[0]
                    
                vol = float(std_val) * 15.87 * 100
                return vol, daily_returns
    except: pass
    return None, None

def update_volatility(start_date):
    vol1, ret1 = calculate_volatility_robust("WBD", start_date)
    vol2, ret2 = calculate_volatility_robust("NFLX", start_date)
    
    # [수정] Pandas Ambiguity Error 방지를 위해 명확한 None 체크로 변경
    if (vol1 is not None) and (vol2 is not None):
        st.session_state['wbd_vol'] = vol1
        st.session_state['nflx_vol'] = vol2
        st.session_state['wbd_returns_data'] = ret1
        st.session_state['nflx_returns_data'] = ret2
        try:
            # 데이터프레임 병합 시 Series 이름 충돌 방지
            r1 = ret1.copy()
            r2 = ret2.copy()
            if isinstance(r1, pd.Series): r1.name = 'WBD'
            if isinstance(r2, pd.Series): r2.name = 'NFLX'
            
            df = pd.concat([r1, r2], axis=1, join='inner')
            st.session_state['correlation'] = df.corr().iloc[0, 1]
        except:
            st.session_state['correlation'] = None
        return True
    return False

@st.cache_data(ttl=30)
def get_live_prices():
    try:
        session = requests.Session()
        session.verify = False
        headers = {'User-Agent': 'Mozilla/5.0'}
        w_data = session.get("https://query1.finance.yahoo.com/v8/finance/chart/WBD", headers=headers, timeout=5).json()
        n_data = session.get("https://query1.finance.yahoo.com/v8/finance/chart/NFLX", headers=headers, timeout=5).json()
        curr_wbd = w_data['chart']['result'][0]['meta']['regularMarketPrice']
        curr_nflx = n_data['chart']['result'][0]['meta']['regularMarketPrice']
        
        # [기능 유지] 서버 위치와 상관없이 '한국 시간(Asia/Seoul)' 강제 적용
        kst = pytz.timezone('Asia/Seoul')
        now_time = datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")
        
        return curr_wbd, curr_nflx, "API-Direct", now_time
    except:
        return None, None, "Fail", None

# ---------------------------------------------------------
# [5] 메인 UI
# ---------------------------------------------------------
st.title("🎬 WBD-NFLX 차익거래 & 헷지 시뮬레이터 (Last Updated: 25.12.24)")
st.markdown("---")

# 자동 업데이트 로직 유지
if st.session_state['nflx_vol'] == 29.5 and st.session_state['wbd_vol'] == 49.0:
    update_volatility(datetime.date(2025, 12, 4))

menu = st.radio("👇 메뉴 선택", ["📉 시나리오 분석", "🎲 몬테카를로", "📊 변동성 상세"], horizontal=True, label_visibility="collapsed")

# 사이드바 설정 계속
st.sidebar.markdown("---")
st.sidebar.header("🎛️ 딜 조건 설정")
target_entry = st.sidebar.number_input("목표 진입가 ($)", value=27.00, step=0.1)
deal_price = 30.00

# 자본금 입력 (WBD 기준)
wbd_input_capital = st.sidebar.number_input("WBD 투자금액 ($)", value=10000, step=1000)

st.sidebar.caption("💡 WBD 포지션을 입력하면 헷지 규모(숏)는 자동 산출됩니다.")
st.sidebar.info(f"📅 현재 시점: {SIMULATED_TODAY}\n\n🎯 공개매수 종료일: {TARGET_DATE}")

curr_wbd, curr_nflx, method, check_time = get_live_prices()
st.sidebar.markdown("---")
if st.sidebar.button("🔄 주가 새로고침"):
    st.cache_data.clear()
    update_volatility(datetime.date(2025, 12, 4))

if curr_wbd is None:
    st.error("❌ 가격 수집 실패 → 수동 입력")
    c1, c2 = st.columns(2)
    curr_wbd = c1.number_input("WBD ($)", value=28.89)
    curr_nflx = c2.number_input("NFLX ($)", value=930.00)
else:
    # 한국 시간임을 명시하기 위해 (KST) 문구 추가
    st.success(f"✅ 실시간 데이터 수신 성공 (방법: {method}) | 🕒 기준 시간: {check_time} (KST)")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("WBD 현재가", f"${curr_wbd:.2f}")
col2.metric("NFLX 현재가", f"${curr_nflx:.2f}")

# 영업일 계산 (동적 날짜 반영)
days_remaining = np.busday_count(SIMULATED_TODAY, TARGET_DATE)
if days_remaining < 0:
    st.error("⚠️ 공개매수 종료일이 지났습니다.")
    days_remaining = 0

T_years = max(days_remaining / 252.0, 0.001)

wbd_vol = st.session_state['wbd_vol']
nflx_vol = st.session_state['nflx_vol']

hedge_ratio = wbd_vol / nflx_vol 

# 총 자본금(Total Capital) 자동 계산
wbd_shares = wbd_input_capital / target_entry 
wbd_total_amt = wbd_shares * target_entry 
nflx_short_amt = wbd_total_amt * hedge_ratio 
nflx_short_shares = nflx_short_amt / curr_nflx
total_real_capital = wbd_total_amt + nflx_short_amt # (WBD 매수 + NFLX 숏)

col3.metric("남은 영업일", f"{days_remaining}일")
col4.metric("헷지 비율", f"{hedge_ratio:.2f}배")
col5.metric("NFLX 숏 규모", f"${nflx_short_amt:,.0f}")
st.markdown("---")

# ---------------------------------------------------------
# [화면 1] 시나리오
# ---------------------------------------------------------
if menu == "📉 시나리오 분석":
    st.subheader("📊 넷플릭스 등락에 따른 손익표")
    
    # 기간 및 연환산 기준 안내 (상단 1회 표시)
    st.info(f"""
    **ℹ️ 수익률 기준 알림 (Investment Period: {INVEST_DAYS} Days)**
    * **투자 기간:** {SIMULATED_TODAY} ~ {TARGET_DATE} (총 {INVEST_DAYS}일)
    * **수익률 기준:** WBD 매수금액뿐만 아니라 **'총 필요 자본(WBD+숏 금액)'**을 기준으로 보수적으로 산출되었습니다.
    * 괄호 안의 **(연 ...%)** 수치는 이를 1년(365일) 기준으로 환산한 수치입니다.
    """)
    
    # 자본금 내역 표시
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 총 필요 자본(Total)", f"${total_real_capital:,.0f}")
    c2.metric("📦 WBD 매수", f"${wbd_total_amt:,.0f}")
    c3.metric("📉 NFLX 숏", f"${nflx_short_amt:,.0f}")
    
    moves = [-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15]
    results = []
    for m in moves:
        total = ((deal_price - target_entry) * wbd_shares) + (-(nflx_short_amt * m))
        
        # 연환산 수익률 계산
        simple_roi = (total / total_real_capital) * 100
        annual_roi = simple_roi * ANNUAL_FACTOR
        
        results.append({
            "NFLX 변동": f"{m*100:+.0f}%", 
            "최종손익($)": round(total), 
            "수익률(%)": f"{simple_roi:.2f}% (연 {annual_roi:.1f}%)" # 문자열 포맷팅
        })
        
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
    
    csv_data = df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 시나리오 결과 다운로드 (CSV)",
        data=csv_data,
        file_name=f"wbd_scenario_{SIMULATED_TODAY.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # 차트용 데이터는 숫자로 다시 만듦
    df['수익률_숫자'] = df['최종손익($)'] / total_real_capital * 100
    st.plotly_chart(px.bar(df, x="NFLX 변동", y="최종손익($)", color="수익률_숫자", color_continuous_scale="RdBu"), use_container_width=True)

# ---------------------------------------------------------
# [화면 2] 몬테카를로
# ---------------------------------------------------------
elif menu == "🎲 몬테카를로":
    st.subheader(f"🎲 {TARGET_DATE} 넷플릭스 주가 및 수익 예측")
    st.caption(f"ℹ️ 적용된 변동성(Vol): NFLX {nflx_vol:.2f}% (기반 데이터: 12/5 ~ 현재)")
    
    # 몬테카를로 탭에도 상단 안내 1회 표시
    st.info(f"""
    **ℹ️ 수익률 기준 알림 (Investment Period: {INVEST_DAYS} Days)**
    * **투자 기간:** {SIMULATED_TODAY} ~ {TARGET_DATE} (총 {INVEST_DAYS}일)
    * **연환산(Annualized):** 괄호 안의 수치는 28일 수익률을 1년(365일) 기준으로 환산한 값입니다.
    """)
    
    c_m1, c_m2 = st.columns(2)
    c_m1.metric("💰 총 필요 자본(Total)", f"${total_real_capital:,.0f}")
    c_m2.metric("📉 헷지 비율", f"{hedge_ratio:.2f}배")

    if st.button("▶️ 분석 시작 (10,000회)"):
        sims = 10000
        shocks = np.random.normal(0, 1, sims)
        sim_prices = curr_nflx * (1 + shocks * (nflx_vol/100) * np.sqrt(T_years))
        
        mean_price = np.mean(sim_prices)
        st.metric("📉 넷플릭스 예상 주가 (평균)", f"${mean_price:.2f}")
        fig_p = px.histogram(x=sim_prices, nbins=100, title="[1단계] 예상 주가 분포", labels={'x': '예상 주가 ($)'}, color_discrete_sequence=['#E50914'])
        fig_p.add_vline(x=mean_price, line_color="yellow", annotation_text=f"평균 ${mean_price:.2f}")
        st.plotly_chart(fig_p, use_container_width=True)

        profit = ((deal_price - target_entry) * wbd_shares) + ((curr_nflx - sim_prices) * nflx_short_shares)
        
        # 총 자본금(Total) 기준으로 ROI 계산
        roi = (profit / total_real_capital) * 100
        
        st.markdown("---")
        st.markdown("### 💰 [2단계] 최종 수익률 분포")
        
        mean_roi = np.mean(roi)
        mean_roi_annual = mean_roi * ANNUAL_FACTOR # 연환산 평균
        
        win_rate = np.sum(roi > 0)/sims*100
        var_95 = np.percentile(roi, 5)

        c1, c2, c3 = st.columns(3)
        # 평균 수익률 옆에 연환산 병기
        c1.metric("평균 수익률", f"{mean_roi:.2f}% (연 {mean_roi_annual:.1f}%)")
        c2.metric("승률", f"{win_rate:.1f}%")
        c3.metric("VaR (95%)", f"{var_95:.2f}%")
        
        # 승률 설명 문구
        st.info(f"""
        **💡 승률(Win Rate)이란?** 10,000번의 미래 시뮬레이션을 돌렸을 때, **최종 손익이 $0(원금 보전) 이상으로 끝난 횟수의 비율**입니다. 
        (예: 승률 {win_rate:.1f}% = 100번 투자하면 {int(win_rate)}번은 돈을 벌거나 잃지 않고, {100-int(win_rate)}번만 손실을 볼 가능성이 있다는 뜻입니다.)
        """)
        
        # ★★★ [복구 완료] 음의 베타에 대한 중요한 설명 (삭제 금지) ★★★
        st.warning("""
        ⚠️ **중요: 실제 수익률은 이보다 높을 가능성이 큽니다**
        
        이 몬테카를로 시뮬레이션은 넷플릭스 주가의 **랜덤워크(무작위 움직임)**를 가정합니다. 
        하지만 실제로 WBD와 NFLX는 **음의 상관관계(베타 < 0)**를 보이고 있습니다.
        
        즉, WBD가 $27→$30으로 상승할 때, NFLX는 **평균적으로 하락**할 가능성이 높습니다.
        따라서 NFLX 숏 포지션에서 **추가 수익**이 발생할 확률이 높아, 
        **실제 예상 수익률은 위 결과보다 더 높을 것으로 예상됩니다.**
        
        💡 두 종목 간 상관관계는 **"📊 변동성 상세"** 탭에서 확인하실 수 있습니다.
        """)
        
        fig_r = px.histogram(x=roi, nbins=100, title="수익률 분포", labels={'x': '수익률 (%)'}, color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_r, use_container_width=True)

# ---------------------------------------------------------
# [화면 3] 변동성 상세 (기능 100% 유지)
# ---------------------------------------------------------
elif menu == "📊 변동성 상세":
    st.subheader("📈 변동성 데이터 관리")
    
    col_c1, col_c2, col_c3 = st.columns([1, 1, 2])
    vol_start = col_c1.date_input("변동성 계산 시작일 (기준일)", value=datetime.date(2025, 12, 4))
    
    if st.session_state['wbd_returns_data'] is None:
         st.info("ℹ️ 데이터 로딩 중...")
    
    if col_c2.button("🔄 변동성 갱신"):
        st.cache_data.clear()
        with st.spinner(f"최신 시장 데이터 분석 중..."):
            if update_volatility(vol_start):
                st.success(f"✅ 갱신 완료! ({vol_start} 이후 데이터만 사용)")
            else:
                st.error("❌ 데이터 부족 또는 실패")
                
    st.markdown("---")
    
    k1, k2, k3 = st.columns(3)
    k1.metric("WBD 변동성 (연)", f"{st.session_state['wbd_vol']:.2f}%")
    k2.metric("NFLX 변동성 (연)", f"{st.session_state['nflx_vol']:.2f}%")
    
    corr_val = st.session_state['correlation']
    if corr_val is not None:
        k3.metric("상관계수", f"{corr_val:.3f}")

    if st.session_state['wbd_returns_data'] is not None:
        ret1 = st.session_state['wbd_returns_data']
        ret2 = st.session_state['nflx_returns_data']
        try:
            df = pd.concat([ret1, ret2], axis=1, join='inner')
            df.columns = ['WBD', 'NFLX']
            
            st.markdown("#### 📅 1. 일별 수익률 데이터 (단순 등락률)")
            table_placeholder = st.empty()
            
            st.markdown("#### 🔗 2. 상관관계 점도표 (Click to Highlight)")
            st.info("💡 **Tip:** 차트의 점을 **클릭**하면, 위 표에서 해당 날짜가 **노란색**으로 표시됩니다.")
            
            x = df['NFLX'] * 100
            y = df['WBD'] * 100
            
            fig = px.scatter(df*100, x='NFLX', y='WBD', hover_data={'NFLX':':.2f', 'WBD':':.2f'})
            if len(df) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                x_range = np.linspace(x.min(), x.max(), 100)
                y_range = slope * x_range + intercept
                fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='추세선', line=dict(color='red', dash='dash')))
            
            selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
            
            st.markdown("#### 📝 3. 분석 결과")
            explanation = ""
            if corr_val > 0.5: explanation = "두 종목이 **강하게 같은 방향**으로 움직입니다. 헷지 효율이 떨어질 수 있습니다."
            elif 0.1 <= corr_val <= 0.5: explanation = "두 종목이 **약하게 같은 방향**으로 움직이는 경향이 있습니다."
            elif -0.1 < corr_val < 0.1: explanation = "두 종목은 **서로 상관없이** 따로 움직입니다. (변동성 헷지에 이상적)"
            else: explanation = "두 종목이 **반대 방향**으로 움직이는 경향이 있습니다."
            st.info(f"**💡 Insight:**\n\n현재 상관계수는 **{corr_val:.3f}**입니다.\n{explanation}")
            
            selected_indices = []
            if selection and "selection" in selection and selection["selection"]["points"]:
                selected_indices = [p["point_index"] for p in selection["selection"]["points"]]
            
            def highlight_selected_rows(row):
                if selected_indices:
                    target_dates = df.iloc[selected_indices].index
                    if row.name in target_dates:
                        return ['background-color: #FFFF00; color: black'] * len(row)
                return [''] * len(row)

            table_placeholder.dataframe(
                (df*100).style.format("{:.2f}%").apply(highlight_selected_rows, axis=1), 
                use_container_width=True
            )
            
        except Exception as e:
            st.warning(f"데이터 표시 중 오류: {e}")
