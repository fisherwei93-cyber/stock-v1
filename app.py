import streamlit as st
import os
import datetime
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re 

# ================= 1. 铁律配置 =================
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if key in os.environ:
        del os.environ[key]

ICON_URL = "https://cdn-icons-png.flaticon.com/512/10452/10452449.png"

st.set_page_config(page_title="摩根·V1 (Map)", layout="wide", page_icon="🦁")

st.markdown(f"""
<head>
    <link rel="apple-touch-icon" href="{ICON_URL}">
    <link rel="icon" type="image/png" href="{ICON_URL}">
</head>
<style>
    /* 全局配置 */
    .stApp {{ background-color: #000000 !important; color: #FFFFFF !important; }}
    section[data-testid="stSidebar"] {{ background-color: #111111 !important; }}
    header {{ visibility: visible !important; }}

    /* 核心高亮 */
    div[data-testid="stMetricValue"] {{
        color: #FFFFFF !important; 
        font-size: 28px !important;
        font-weight: 900 !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }}
    div[data-testid="stMetricLabel"] {{ color: #9CA3AF !important; font-weight: 700 !important; }}
    
    /* 侧边栏财报卡片 */
    .earning-card {{
        background: #1e1b4b; 
        border-left: 4px solid #6366f1;
        padding: 8px;
        margin-bottom: 6px;
        border-radius: 4px;
    }}
    .earning-alert {{
        background: #450a0a;
        border-left: 4px solid #ef4444;
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }}
        70% {{ box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
    }}
    .ec-row {{ display: flex; justify-content: space-between; align-items: center; font-size: 13px; }}
    .ec-ticker {{ font-weight: bold; color: #fff; }}
    .ec-date {{ color: #cbd5e1; font-family: monospace; }}
    .ec-sector {{ font-size: 10px; padding: 1px 4px; border-radius: 3px; background: #333; color: #aaa; margin-top: 4px; display: inline-block;}}

    /* 视野黄框 */
    .l-box {{
        background-color: #FF9F1C; color: #000000 !important; padding: 15px;
        border-radius: 8px; margin-bottom: 20px;
    }}
    .l-title {{ font-size: 18px; font-weight: 900; border-bottom: 2px solid #000; padding-bottom: 8px; margin-bottom: 12px; color: #000; }}
    .l-item {{ display: flex; justify-content: space-between; border-bottom: 1px dashed rgba(0,0,0,0.2); padding: 4px 0; color: #000; font-weight: 600; }}
    
    /* 组件样式 */
    .score-card {{ background: #1A1A1A; padding: 15px; border-radius: 12px; text-align: center; border: 1px solid #333; margin-bottom: 15px; }}
    .sc-val {{ font-size: 42px; font-weight: 900; color: #4ade80; line-height: 1; }}
    .sc-lbl {{ font-size: 12px; color: #D1D5DB; font-weight: bold; }}
    .wl-row {{ background-color: #1A1A1A; padding: 12px; margin-bottom: 8px; border-radius: 6px; border-left: 4px solid #555; display: flex; justify-content: space-between; align-items: center; color: #FFFFFF; }}
    
    /* 研报样式 */
    .report-title {{ font-size: 22px; font-weight: 900; color: #FF9F1C; margin-bottom: 10px; border-left: 5px solid #FF9F1C; padding-left: 10px; }}
    .report-text {{ font-size: 15px; line-height: 1.8; color: #E5E7EB; margin-bottom: 20px; background: #1A1A1A; padding: 15px; border-radius: 8px; }}
    .guru-check {{ display: flex; align-items: center; margin-bottom: 8px; padding: 8px; background: #262626; border-radius: 6px; }}
</style>
""", unsafe_allow_html=True)

import yfinance as yf

# ================= 2. 全行业财报引擎 =================

# [NEW] 全行业财报地图 (12小时缓存)
@st.cache_data(ttl=43200, show_spinner=False)
def fetch_sector_earnings():
    # 30只全行业龙头
    sectors = {
        "💻 科技七巨头": ["NVDA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA"],
        "🏦 金融/支付": ["JPM", "BAC", "V", "COIN", "BLK"],
        "💊 医药/消费": ["LLY", "JNJ", "PG", "KO", "MCD", "NVO"],
        "⛽ 能源/工业": ["XOM", "CVX", "CAT", "GE", "LMT"],
        "💎 芯片/硬件": ["AMD", "AVGO", "TSM", "QCOM", "ASML"]
    }
    
    flat_list = []
    for sec, tickers in sectors.items():
        for t in tickers: flat_list.append((t, sec))
    
    results = []
    today = datetime.date.today()
    
    # 批量获取太慢，这里用循环 (后台静默执行)
    for t, sec in flat_list:
        try:
            s = yf.Ticker(t)
            cal = s.calendar
            e_date = None
            
            # 兼容性读取
            if isinstance(cal, dict) and cal:
                if 'Earnings Date' in cal: e_date = cal['Earnings Date'][0]
                elif 'Earnings High' in cal: e_date = cal.get('Earnings Date', [])[0]
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                e_date = cal.iloc[0, 0]
                
            if e_date:
                # 统一转为 date 对象
                if isinstance(e_date, (datetime.datetime, pd.Timestamp)):
                    ed = e_date.date()
                else:
                    ed = datetime.datetime.strptime(str(e_date).split()[0], "%Y-%m-%d").date()
                
                if ed >= today:
                    days_left = (ed - today).days
                    results.append({
                        "Code": t, "Sector": sec, "Date": str(ed), 
                        "Days": days_left, "Sort": days_left
                    })
        except: pass
        
    if results:
        return sorted(results, key=lambda x: x['Sort'])
    return []

# ================= 3. 数据引擎 (Core) =================

@st.cache_data(ttl=30, show_spinner=False)
def fetch_realtime_price(ticker):
    try:
        s = yf.Ticker(ticker)
        try:
            price = s.fast_info.last_price
            prev = s.fast_info.previous_close
        except:
            info = s.info if s.info is not None else {}
            price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            prev = info.get('previousClose', price)
        return {"price": price, "prev": prev}
    except: return {"price": 0, "prev": 0}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_heavy_data(ticker):
    import yfinance as yf
    max_retries = 3; h = pd.DataFrame()
    s = yf.Ticker(ticker)
    
    for attempt in range(max_retries):
        try:
            h = s.history(period="2y")
            if not h.empty: break
            time.sleep(1)
        except: time.sleep(2**attempt)
            
    if h.empty: return {"history": pd.DataFrame(), "info": {}, "error": "No Data"}

    # 指标计算 (V88 Core)
    h['MA20'] = h['Close'].rolling(20).mean()
    h['MA200'] = h['Close'].rolling(200).mean()
    
    # SuperTrend
    h['TR'] = np.maximum(h['High'] - h['Low'], np.abs(h['High'] - h['Close'].shift(1)))
    h['ATR'] = h['TR'].rolling(10).mean()
    h['ST_Lower'] = ((h['High']+h['Low'])/2) - (3 * h['ATR'])
    
    # VWAP & FVG
    v = h['Volume'].values; tp = (h['High'] + h['Low'] + h['Close']) / 3
    h['VWAP'] = (tp * v).cumsum() / v.cumsum()
    h['FVG_Bull'] = (h['Low'] > h['High'].shift(2))
    
    # Z-Score & HMA
    h['STD20'] = h['Close'].rolling(20).std()
    h['Z_Score'] = (h['Close'] - h['MA20']) / h['STD20']
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    period = 14
    wma_half = wma(h['Close'], period // 2)
    wma_full = wma(h['Close'], period)
    h['HMA'] = wma(2 * wma_half - wma_full, int(np.sqrt(period)))
    
    # ADX / Ichimoku / CCI / WR / CMF / MACD / RSI
    plus_dm = h['High'].diff(); minus_dm = h['Low'].diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm > 0] = 0; minus_dm = minus_dm.abs()
    tr14 = h['TR'].rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    h['ADX'] = dx.rolling(14).mean()
    
    h['Tenkan'] = (h['High'].rolling(9).max() + h['Low'].rolling(9).min()) / 2
    h['Kijun'] = (h['High'].rolling(26).max() + h['Low'].rolling(26).min()) / 2
    h['SpanA'] = ((h['Tenkan'] + h['Kijun']) / 2).shift(26)
    h['SpanB'] = ((h['High'].rolling(52).max() + h['Low'].rolling(52).min()) / 2).shift(26)
    
    sma_tp = tp.rolling(20).mean()
    def calc_mad(x): return np.mean(np.abs(x - np.mean(x)))
    mad = tp.rolling(20).apply(calc_mad, raw=True)
    h['CCI'] = (tp - sma_tp) / (0.015 * mad)
    
    hh = h['High'].rolling(14).max(); ll = h['Low'].rolling(14).min()
    h['WR'] = -100 * (hh - h['Close']) / (hh - ll)
    
    mfm = ((h['Close'] - h['Low']) - (h['High'] - h['Close'])) / (h['High'] - h['Low'])
    mfv = mfm * h['Volume']
    h['CMF'] = mfv.rolling(20).sum() / h['Volume'].rolling(20).sum()
    
    exp12 = h['Close'].ewm(span=12).mean(); exp26 = h['Close'].ewm(span=26).mean()
    h['MACD'] = exp12 - exp26; h['Signal'] = h['MACD'].ewm(span=9).mean()
    
    delta = h['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss; h['RSI'] = 100 - (100 / (1 + rs))
    h['UPPER'] = h['MA20'] + 2*h['STD20']; h['LOWER'] = h['MA20'] - 2*h['STD20']
    
    h['DC_Upper'] = h['High'].rolling(20).max(); h['DC_Lower'] = h['Low'].rolling(20).min()

    safe_info = s.info if s.info is not None else {}
    return {"history": h, "info": safe_info, "error": None}

# 辅助函数
def fmt_pct(v): return f"{v:.2%}" if isinstance(v, (int, float)) else "-"
def fmt_num(v): return f"{v:.2f}" if isinstance(v, (int, float)) else "-"
def fmt_big(v): 
    if not isinstance(v, (int, float)): return "-"
    if v > 1e12: return f"{v/1e12:.2f}T"
    if v > 1e9: return f"{v/1e9:.2f}B"
    return str(v)
def smart_translate(t, d): 
    if not isinstance(t, str): return t
    for k,v in d.items(): 
        if k.lower() in t.lower(): return v
    return t

# ================= 4. 业务逻辑 =================
def process_news(news_list):
    import re
    if not news_list: return pd.DataFrame()
    res = []
    pat = r"(\$|USD)\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)|(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s?(美元|USD)"
    for n in news_list:
        title = n.get('title', 'No Title'); link = n.get('link', '#')
        match = re.search(pat, title)
        price = "-"
        if match:
            vals = [g for g in match.groups() if g and g not in ['$','USD','美元']]
            if vals: price = f"${vals[0]}"
        ts = n.get('providerPublishTime', 0)
        t_str = pd.to_datetime(ts, unit='s').strftime('%m-%d %H:%M')
        res.append({"时间": t_str, "标题": title, "价格": price, "链接": link})
    return pd.DataFrame(res)

def calculate_quant_score(info, history):
    if not isinstance(info, dict): info = {}
    score = 50; notes = []
    if not history.empty:
        c = history['Close'].iloc[-1]; ma50 = history['Close'].rolling(50).mean().iloc[-1]
        if c > ma50: score += 15; notes.append("趋势向上")
    pe = info.get('forwardPE')
    if pe and pe < 30: score += 10; notes.append("估值健康")
    gr = info.get('revenueGrowth')
    if gr and gr > 0.15: score += 10; notes.append("高成长")
    rec = info.get('recommendationMean')
    if rec and rec < 2.0: score += 15; notes.append("机构强推")
    return min(100, max(0, int(score))), " | ".join(notes)

def calculate_seasonality(df):
    if df.empty: return None
    df = df.copy()
    df['Month'] = df.index.month
    df['Ret'] = df['Close'].pct_change()
    monthly_stats = df.groupby('Month')['Ret'].agg(['mean', lambda x: (x>0).sum() / len(x)])
    monthly_stats.columns = ['Avg Return', 'Win Rate']
    return monthly_stats

def calculate_volume_profile(df, bins=50):
    price_min = df['Low'].min(); price_max = df['High'].max()
    hist = np.histogram(df['Close'], bins=bins, range=(price_min, price_max), weights=df['Volume'])
    return hist[1][:-1], hist[0]

def calculate_max_pain(calls, puts):
    if calls.empty or puts.empty: return 0
    strikes = sorted(set(calls['strike']).union(set(puts['strike'])))
    min_loss = float('inf'); max_pain = 0
    for s in strikes:
        loss = 0
        c_loss = calls[calls['strike'] < s].apply(lambda x: (s - x['strike']) * x['openInterest'], axis=1).sum()
        p_loss = puts[puts['strike'] > s].apply(lambda x: (x['strike'] - s) * x['openInterest'], axis=1).sum()
        loss = c_loss + p_loss
        if loss < min_loss: min_loss = loss; max_pain = s
    return max_pain

# ================= 5. 主程序 =================
if 'watchlist' not in st.session_state: st.session_state.watchlist = ['TSLA', 'NVDA', 'AAPL', 'AMD', 'PLTR']
if 'current_ticker' not in st.session_state: st.session_state.current_ticker = 'TSLA'

# Sidebar
with st.sidebar:
    st.title("🦁 摩根·V1")
    new_ticker = st.text_input("🔍 搜索", "").upper()
    if new_ticker: st.session_state.current_ticker = new_ticker; st.rerun()

    # [NEW] 全行业财报侧边栏
    st.markdown("---")
    st.caption("📅 财报雷达 (全行业)")
    earnings_list = fetch_sector_earnings()
    if earnings_list:
        # 分组显示：7天内(Urgent) 和 其他
        urgent = [x for x in earnings_list if x['Days'] <= 7]
        later = [x for x in earnings_list if x['Days'] > 7]
        
        if urgent:
            st.markdown("**🚨 近期爆发 (7天内)**")
            for item in urgent:
                st.markdown(f"""
                <div class='earning-card earning-alert'>
                    <div class='ec-row'>
                        <span class='ec-ticker'>{item['Code']}</span>
                        <span class='ec-date'>{item['Date']} (T-{item['Days']})</span>
                    </div>
                    <div class='ec-sector'>{item['Sector']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        if later:
            with st.expander("🔭 未来展望", expanded=False):
                for item in later:
                    st.markdown(f"""
                    <div class='earning-card'>
                        <div class='ec-row'>
                            <span class='ec-ticker'>{item['Code']}</span>
                            <span class='ec-date'>{item['Date']}</span>
                        </div>
                        <div class='ec-sector'>{item['Sector']}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.caption("暂无数据 (后台更新中...)")

    st.markdown("---")
    st.caption("我的自选")
    for t in st.session_state.watchlist:
        p_data = fetch_realtime_price(t)
        chg = (p_data['price'] - p_data['prev']) / p_data['prev'] if p_data['prev'] else 0
        c_color = "#4ade80" if chg >= 0 else "#f87171"
        c1, c2 = st.columns([2, 1])
        if c1.button(f"{t}", key=f"btn_{t}"):
            st.session_state.current_ticker = t; st.rerun()
        c2.markdown(f"<span style='color:{c_color}'>{chg:.2%}</span>", unsafe_allow_html=True)

# Main
page = st.sidebar.radio("📌 导航", ["🚀 股票分析", "🗓️ 财报地图", "📖 功能说明书"])

if page == "🚀 股票分析":
    ticker = st.session_state.current_ticker
    
    # 1. 实时报价
    p_data = fetch_realtime_price(ticker)
    p = p_data['price']; prev = p_data['prev']
    chg_val = p - prev; chg_pct = chg_val / prev if prev else 0
    color = "#4ade80" if chg_val >= 0 else "#f87171"
    bg = "rgba(74, 222, 128, 0.1)" if chg_val >= 0 else "rgba(248, 113, 113, 0.1)"

    st.markdown(f"""
    <div class='price-container'>
        <div style='color:#9CA3AF; font-size:14px; font-weight:bold; letter-spacing:1px;'>{ticker} 实时报价</div>
        <div class='big-price' style='color:{color}'>${p:.2f}</div>
        <div class='price-change' style='background-color:{bg}; color:{color}'>
            {chg_val:+.2f} ({chg_pct:+.2%})
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c_btn = st.columns(4)
    c_btn[0].link_button("🔥 谷歌", f"https://www.google.com/search?q=why+is+{ticker}+moving")
    c_btn[1].link_button("🎯 目标价", f"https://www.google.com/search?q={ticker}+stock+target")
    c_btn[2].link_button("👽 Reddit", f"https://www.reddit.com/search/?q=${ticker}")
    c_btn[3].link_button("🐦 Twitter", f"https://twitter.com/search?q=${ticker}")

    # 2. 深度数据
    with st.spinner("🦁 正在调取机构底仓数据..."):
        heavy = fetch_heavy_data(ticker)

    if heavy['error']:
        st.warning(f"深度数据暂时不可用: {heavy['error']}")
        h, i = pd.DataFrame(), {}
    else:
        h, i = heavy['history'], heavy['info']

    # [FIX V92] 强制初始化变量，防止 NameError
    rt_price = p if p > 0 else (h['Close'].iloc[-1] if not h.empty else 0)

    if not h.empty:
        curr = h['Close'].iloc[-1]
        ma20 = h['MA20'].iloc[-1]; ma200 = h['MA200'].iloc[-1]
        res = h['High'].tail(20).max(); sup = h['Low'].tail(20).min()
        
        st.markdown(f"""
        <div class='l-box'>
            <div class='l-title'>🦁 视野·交易计划 ({ticker})</div>
            <div class='l-item'><span>压力位 (R1)</span><span style='color:#f87171'>${res:.2f}</span></div>
            <div class='l-item'><span>支撑位 (S1)</span><span style='color:#4ade80'>${sup:.2f}</span></div>
            <div class='l-item'><span>趋势判断</span><span>{'🐂 牛市' if curr > ma200 else '🐻 熊市'}</span></div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📈 机构趋势图 (SuperTrend)", expanded=True):
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'], name='K线'))
            fig.add_trace(go.Scatter(x=h.index, y=h['ST_Lower'], mode='markers', marker=dict(color='orange', size=2), name='止损线'))
            fig.add_trace(go.Scatter(x=h.index, y=h['VWAP'], line=dict(color='#fcd34d', width=1), name='VWAP'))
            fig.add_trace(go.Scatter(x=h.index, y=h['HMA'], line=dict(color='#ec4899', width=1), name='HMA'))
            for idx in range(len(h)-50, len(h)): 
                if h['FVG_Bull'].iloc[idx]: fig.add_shape(type="rect", x0=h.index[idx-2], y0=h['Low'].iloc[idx], x1=h.index[idx], y1=h['High'].iloc[idx-2], fillcolor="rgba(139, 92, 246, 0.3)", line_width=0)
            fig.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    # 3. 标签页功能
    st.session_state.quant_score = calculate_quant_score(i, h)
    tabs = st.tabs(["📰 资讯", "👥 持仓", "💰 估值", "🎓 深度研报", "📉 进阶指标"])

    with tabs[0]:
        news_df = process_news(heavy.get('news', []))
        if not news_df.empty: st.dataframe(news_df[['时间','标题','价格','链接']], column_config={"链接": st.column_config.LinkColumn("阅读"), "价格": st.column_config.TextColumn("🎯", width="small")}, hide_index=True)
        else: st.info("暂无新闻")
        
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🏦 机构持仓")
            if heavy.get('inst') is not None: st.dataframe(heavy['inst'], use_container_width=True)
        with c2:
            st.subheader("🕴️ 内部交易")
            if heavy.get('insider') is not None: st.dataframe(heavy['insider'], use_container_width=True)

    with tabs[2]:
        st.subheader("⚖️ 格雷厄姆合理价")
        safe_i = i if isinstance(i, dict) else {}
        eps = safe_i.get('trailingEps', 0); bvps = safe_i.get('bookValue', 0)
        
        # [FIX] 崩溃修复点：增加 rt_price > 0 的判断
        if eps is not None and bvps is not None and eps > 0 and bvps > 0 and rt_price > 0:
            graham = (22.5 * eps * bvps) ** 0.5
            upside = (graham - rt_price) / rt_price
            st.metric("Graham Number", f"${graham:.2f}", f"{upside:.1%} Upside")
        else: 
            st.error("数据不足 (EPS/BVPS缺失 或 价格为0)")

        st.markdown("---")
        st.subheader("💰 DCF 模型")
        peg = safe_i.get('pegRatio')
        if peg: st.caption(f"PEG: {peg} {'✅' if peg < 1 else '⚠️'}")
        g = st.slider("预期增长率 %", 0, 50, 15)
        if eps is not None and eps > 0:
            val = (eps * ((1+g/100)**5) * 25) / (1.1**5)
            st.metric("估值", f"${val:.2f}")

    with tabs[3]: # 深度研报
        st.header(f"🎓 {ticker} 深度研报")
        st.markdown("<div class='report-title'>1. 🏢 商业模式</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='report-text'>{i.get('longBusinessSummary', '暂无描述')}</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='report-title'>2. 🏰 护城河分析</div>", unsafe_allow_html=True)
        gm = i.get('grossMargins', 0); roe = i.get('returnOnEquity', 0)
        c_m1, c_m2 = st.columns(2)
        c_m1.markdown(f"<div class='score-card'><div class='sc-lbl'>毛利率</div><div class='sc-val' style='color:{'#4ade80' if gm>0.4 else '#f87171'}'>{fmt_pct(gm)}</div><div class='sc-lbl'>标准: >40%</div></div>", unsafe_allow_html=True)
        c_m2.markdown(f"<div class='score-card'><div class='sc-lbl'>ROE</div><div class='sc-val' style='color:{'#4ade80' if roe>0.15 else '#f87171'}'>{fmt_pct(roe)}</div><div class='sc-lbl'>标准: >15%</div></div>", unsafe_allow_html=True)
        
        st.markdown("<div class='report-title'>3. 📞 尽职调查</div>", unsafe_allow_html=True)
        dd1, dd2 = st.columns(2)
        dd1.link_button("📄 SEC 10-K", f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}")
        dd2.link_button("🗣️ Earnings Call", f"https://www.google.com/search?q={ticker}+earnings+call+transcript")

    with tabs[4]: # 进阶指标
        if not h.empty:
            vp_price, vp_vol = calculate_volume_profile(h.iloc[-252:])
            fig3 = make_subplots(rows=4, cols=2, shared_xaxes=True, row_heights=[0.25]*4, column_widths=[0.85, 0.15], specs=[[{"colspan":2}, None], [{"colspan":2}, None], [{"colspan":2}, None], [{"colspan":1}, {}]])
            fig3.add_trace(go.Scatter(x=h.index, y=h['Z_Score'], line=dict(color='#f472b6', width=1), name='Z-Score'), row=1, col=1)
            fig3.add_hline(y=2, line_dash='dot', row=1, col=1); fig3.add_hline(y=-2, line_dash='dot', row=1, col=1)
            fig3.add_trace(go.Scatter(x=h.index, y=h['ADX'], line=dict(color='#fbbf24', width=1), name='ADX'), row=2, col=1)
            fig3.add_hline(y=25, line_dash='dot', row=2, col=1)
            fig3.add_trace(go.Scatter(x=h.index, y=h['CCI'], line=dict(color='#22d3ee', width=1), name='CCI'), row=3, col=1)
            fig3.add_hline(y=100, line_dash='dot', row=3, col=1); fig3.add_hline(y=-100, line_dash='dot', row=3, col=1)
            fig3.add_trace(go.Scatter(x=h.index, y=h['Close'], line=dict(color='#3b82f6', width=1), name='Close'), row=4, col=1)
            fig3.add_trace(go.Bar(x=vp_vol, y=vp_price, orientation='h', marker_color='rgba(100,100,100,0.3)', name='Profile'), row=4, col=2)
            fig3.update_layout(height=800, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig3, use_container_width=True)

elif page == "🗓️ 财报地图":
    st.title("🗓️ 全行业财报热力图")
    st.caption("30+ 明星股财报时间表 (按倒计时排序)")
    
    data = fetch_sector_earnings()
    if data:
        # 转为 DataFrame 并美化
        df = pd.DataFrame(data)
        
        # 1. 7天内高亮区
        urgent = df[df['Days'] <= 7]
        if not urgent.empty:
            st.error("🚨 红色警戒：未来 7 天内发财报的巨头！")
            st.dataframe(urgent[['Code', 'Sector', 'Date', 'Days']].set_index('Code'), use_container_width=True)
        
        # 2. 完整热力表 (Color coded by Days)
        st.subheader("🔭 完整观察名单")
        fig = px.treemap(df, path=[px.Constant("全市场"), 'Sector', 'Code'], values=np.ones(len(df)),
                         color='Days', color_continuous_scale='RdYlGn',
                         hover_data=['Date', 'Days'],
                         title="财报热力分布 (颜色越红 = 距离越近)")
        fig.update_layout(height=500, template="plotly_dark", margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. 详细表格
        with st.expander("查看详细时间表"):
            st.dataframe(df[['Code', 'Sector', 'Date', 'Days']].set_index('Code'), use_container_width=True)
    else:
        st.info("数据正在后台同步中，请稍后刷新...")

else:
    # 渲染维基百科
    st.title("📚 摩根·功能说明书 (Wiki)")
    st.markdown("""
    <div class='wiki-card'><div class='wiki-title'>1. 视野·交易计划 (Vision L-Box)</div><div class='wiki-text'><b>核心逻辑：</b> L战法系统。<br><b>黄框</b>：系统大脑。<br><span class='wiki-tag'>R1/R2</span> 压力位。<br><span class='wiki-tag'>S1/S2</span> 支撑位。</div></div>
    <div class='wiki-card'><div class='wiki-title'>2. 神奇九转 (TD Sequential)</div><div class='wiki-text'><b>原理：</b> 寻找衰竭点。<br><span style='color:#f87171'><b>红色 9</b></span>：上涨力竭(卖)。<br><span style='color:#4ade80'><b>绿色 9</b></span>：下跌力竭(买)。</div></div>
    <div class='wiki-card'><div class='wiki-title'>3. VWAP (机构线)</div><div class='wiki-text'><b>原理：</b> 机构持仓成本。<br>股价 > VWAP：机构护盘。<br>股价 < VWAP：机构出货。</div></div>
    <div class='wiki-card'><div class='wiki-title'>4. 蒙特卡洛预测 (Monte Carlo)</div><div class='wiki-text'><b>原理：</b> 模拟未来30天100种走势。<br><b>悲观底线</b>：95%概率不跌破的止损位。</div></div>
    <div class='wiki-card'><div class='wiki-title'>5. 六维雷达 (Spider)</div><div class='wiki-text'><b>原理：</b> 公司体检表。面积越大，基本面越完美。</div></div>
    <div class='wiki-card'><div class='wiki-title'>6. SuperTrend</div><div class='wiki-text'><b>原理：</b> 趋势跟踪。<b>绿色</b>持有，<b>红色</b>空仓。</div></div>
    <div class='wiki-card'><div class='wiki-title'>7. FVG (缺口)</div><div class='wiki-text'><b>原理：</b> 机构暴力拉升留下的<b>紫色方块</b>。股价常会回调填补。</div></div>
    <div class='wiki-card'><div class='wiki-title'>8. Z-Score (乖离)</div><div class='wiki-text'><b>原理：</b> 统计学偏差。<br>>2: 涨过头(回调风险) <br><-2: 跌过头(反弹机会)。</div></div>
    """, unsafe_allow_html=True)