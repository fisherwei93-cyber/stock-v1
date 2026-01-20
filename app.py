import streamlit as st
import os
import datetime
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re # [FIX] 显式全局导入

# ================= 1. 铁律配置 =================
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if key in os.environ:
        del os.environ[key]

ICON_URL = "https://cdn-icons-png.flaticon.com/512/10452/10452449.png"

st.set_page_config(page_title="摩根·V1 (Live)", layout="wide", page_icon="🦁")

st.markdown(f"""
<head>
    <link rel="apple-touch-icon" href="{ICON_URL}">
    <link rel="icon" type="image/png" href="{ICON_URL}">
</head>
<style>
    .stApp {{ background-color: #000000 !important; color: #FFFFFF !important; }}
    section[data-testid="stSidebar"] {{ background-color: #111111 !important; }}
    header {{ visibility: visible !important; }}

    div[data-testid="stMetricValue"] {{
        color: #FFFFFF !important; 
        font-size: 28px !important;
        font-weight: 900 !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }}
    div[data-testid="stMetricLabel"] {{
        color: #9CA3AF !important;
        font-size: 14px !important;
        font-weight: 700 !important;
    }}
    
    .streamlit-expanderHeader {{
        background-color: #222222 !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }}
    .streamlit-expanderHeader p {{
        color: #FFFFFF !important;
        font-size: 16px !important;
        font-weight: 700 !important;
    }}
    .streamlit-expanderHeader:hover {{
        border-color: #FF9F1C !important;
        color: #FF9F1C !important;
    }}

    .l-box {{
        background-color: #FF9F1C;
        color: #000000 !important;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(255, 159, 28, 0.4);
        margin-bottom: 20px;
        border: 1px solid #e68a00;
        font-family: 'Segoe UI', sans-serif;
    }}
    .l-title {{ font-size: 18px; font-weight: 900; border-bottom: 2px solid #000; padding-bottom: 8px; margin-bottom: 12px; color: #000; }}
    .l-item {{ display: flex; justify-content: space-between; align-items: center; font-size: 14px; font-weight: 600; border-bottom: 1px dashed rgba(0,0,0,0.2); padding: 4px 0; color: #000; }}
    
    .tg-s {{ background: rgba(0,0,0,0.1); padding: 1px 5px; border-radius: 4px; font-size: 11px; margin-left: 6px; color: #333; }}
    .tg-m {{ background: #fffbeb; padding: 1px 5px; border-radius: 4px; font-size: 11px; margin-left: 6px; color: #854d0e; border: 1px solid #eab308; }}
    .tg-h {{ background: #000; color: #FF9F1C; padding: 1px 6px; border-radius: 4px; font-size: 11px; margin-left: 6px; font-weight: 800; }}
    
    .score-card {{ background: #1A1A1A; padding: 15px; border-radius: 12px; text-align: center; border: 1px solid #333; margin-bottom: 15px; }}
    .sc-val {{ font-size: 42px; font-weight: 900; color: #4ade80; line-height: 1; }}
    .sc-lbl {{ font-size: 12px; color: #D1D5DB; font-weight: bold; }}
    
    .wl-row {{ background-color: #1A1A1A; padding: 12px; margin-bottom: 8px; border-radius: 6px; border-left: 4px solid #555; cursor: pointer; display: flex; justify-content: space-between; align-items: center; border: 1px solid #333; color: #FFFFFF; }}
    .wl-row:hover {{ border-left-color: #FF9F1C; background-color: #2A2A2A; }}
    
    .social-box {{ display: flex; gap: 10px; margin-top: 10px; }}
    .sig-box {{ background: rgba(6, 78, 59, 0.8); border: 1px solid #065f46; padding: 10px; border-radius: 6px; margin-top: 10px; font-size: 13px; color: #fff; }}
    .risk-box {{ background: rgba(127, 29, 29, 0.5); border: 1px solid #ef4444; padding: 10px; border-radius: 6px; margin-top: 10px; font-size: 13px; color: #fff; }}
    .note-box {{ background: #1e1b4b; border-left: 4px solid #6366f1; padding: 10px; font-size: 12px; color: #e0e7ff; margin-top: 5px; border-radius: 4px; line-height: 1.6; }}
    .teach-box {{ background: #422006; border-left: 4px solid #f97316; padding: 10px; font-size: 12px; color: #ffedd5; margin-top: 10px; border-radius: 4px; }}
    .mc-box {{ background: #0f172a; border: 1px solid #1e293b; padding: 10px; border-radius: 6px; margin-top:5px; }}
    
    .thesis-col {{ flex: 1; padding: 10px; border-radius: 6px; font-size: 13px; margin-top:5px; }}
    .thesis-bull {{ background: rgba(6, 78, 59, 0.8); border: 1px solid #34d399; color: #fff; }}
    .thesis-bear {{ background: rgba(127, 29, 29, 0.8); border: 1px solid #f87171; color: #fff; }}
    
    .report-title {{ font-size: 22px; font-weight: 900; color: #FF9F1C; margin-bottom: 10px; border-left: 5px solid #FF9F1C; padding-left: 10px; }}
    .report-text {{ font-size: 15px; line-height: 1.8; color: #E5E7EB; margin-bottom: 20px; background: #1A1A1A; padding: 15px; border-radius: 8px; }}
    .guru-check {{ display: flex; align-items: center; margin-bottom: 8px; padding: 8px; background: #262626; border-radius: 6px; }}
    
    .wiki-card {{ background: #1A1A1A; border: 1px solid #333; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
    .wiki-title {{ font-size: 20px; font-weight: bold; color: #FF9F1C; margin-bottom: 15px; border-bottom: 1px solid #444; padding-bottom: 5px; }}
    .wiki-text {{ font-size: 14px; color: #E5E7EB; line-height: 1.8; margin-bottom: 10px; }}
    .wiki-tag {{ background: #374151; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 5px; border: 1px solid #555; }}
    
    /* 核心报价盘 */
    .price-container {{
        background: #1A1A1A; padding: 20px; border-radius: 15px; border: 1px solid #333;
        text-align: center; margin-bottom: 20px;
    }}
    .big-price {{
        font-size: 56px !important; font-weight: 900 !important; color: #FFFFFF;
        line-height: 1.1; text-shadow: 0 0 20px rgba(255,255,255,0.1);
    }}
    .price-change {{
        font-size: 24px !important; font-weight: bold; padding: 5px 15px;
        border-radius: 8px; display: inline-block;
    }}
    .ext-price {{ font-size: 16px !important; color: #9CA3AF; margin-top: 8px; font-family: monospace; }}
</style>
""", unsafe_allow_html=True)

import yfinance as yf

# ================= 2. 智能数据引擎 (Fixes & Updates) =================

# 🔴 快通道：实时价格 (30s缓存，平衡防封锁与实时性)
@st.cache_data(ttl=30, show_spinner=False)
def fetch_realtime_price(ticker):
    try:
        s = yf.Ticker(ticker)
        # 尝试 fast_info
        try:
            price = s.fast_info.last_price
            prev = s.fast_info.previous_close
        except:
            # 降级 info
            info = s.info if s.info is not None else {}
            price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            prev = info.get('previousClose', price)
        
        # [NEW] 盘前/盘后逻辑增强
        ext_price = None
        ext_label = ""
        try:
            info = s.info if s.info is not None else {}
            
            # 检查是否有明确的 pre/post 字段
            pm = info.get('preMarketPrice')
            post = info.get('postMarketPrice')
            curr = info.get('currentPrice')
            reg = info.get('regularMarketPrice')

            # 逻辑：如果当前价 != 常规价，说明在盘前盘后
            if curr and reg and abs(curr - reg) > 0.01:
                # 此时 curr 就是盘前/盘后价
                pass 
            
            # 显式检查
            if pm and abs(pm - price) > 0.01:
                ext_price = pm
                ext_label = "盘前 (Pre)"
            elif post and abs(post - price) > 0.01:
                ext_price = post
                ext_label = "盘后 (Post)"
                
        except: pass

        return {"price": price, "prev": prev, "ext_price": ext_price, "ext_label": ext_label}
    except:
        return {"price": 0, "prev": 0, "ext_price": None, "ext_label": ""}

# 🔵 慢通道：深度数据 (1h缓存)
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_heavy_data(ticker):
    import yfinance as yf # 局部导入
    
    # 重试机制
    max_retries = 3
    h = pd.DataFrame()
    s = yf.Ticker(ticker)
    
    for attempt in range(max_retries):
        try:
            h = s.history(period="2y")
            if not h.empty: break
            time.sleep(1)
        except: 
            if attempt == max_retries - 1:
                return {"history": pd.DataFrame(), "info": {}, "error": "Rate Limited"}
            time.sleep(2**attempt)
            
    if h.empty: return {"history": pd.DataFrame(), "info": {}, "error": "No Data"}

    # --- 指标计算 ---
    h['MA20'] = h['Close'].rolling(20).mean()
    h['MA200'] = h['Close'].rolling(200).mean()
    
    # SuperTrend
    h['TR'] = np.maximum(h['High'] - h['Low'], np.abs(h['High'] - h['Close'].shift(1)))
    h['ATR'] = h['TR'].rolling(10).mean()
    h['ST_Lower'] = ((h['High']+h['Low'])/2) - (3 * h['ATR'])
    
    # VWAP
    v = h['Volume'].values
    tp = (h['High'] + h['Low'] + h['Close']) / 3
    h['VWAP'] = (tp * v).cumsum() / v.cumsum()

    # FVG
    h['FVG_Bull'] = (h['Low'] > h['High'].shift(2))
    
    # Z-Score
    h['STD20'] = h['Close'].rolling(20).std()
    h['Z_Score'] = (h['Close'] - h['MA20']) / h['STD20']
    
    # HMA
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    period = 14
    wma_half = wma(h['Close'], period // 2)
    wma_full = wma(h['Close'], period)
    h['HMA'] = wma(2 * wma_half - wma_full, int(np.sqrt(period)))
    
    # ADX
    plus_dm = h['High'].diff(); minus_dm = h['Low'].diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm > 0] = 0; minus_dm = minus_dm.abs()
    tr14 = h['TR'].rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    h['ADX'] = dx.rolling(14).mean()
    
    # Ichimoku
    h['Tenkan_sen'] = (h['High'].rolling(9).max() + h['Low'].rolling(9).min()) / 2
    h['Kijun_sen'] = (h['High'].rolling(26).max() + h['Low'].rolling(26).min()) / 2
    h['Senkou_Span_A'] = ((h['Tenkan_sen'] + h['Kijun_sen']) / 2).shift(26)
    h['Senkou_Span_B'] = ((h['High'].rolling(52).max() + h['Low'].rolling(52).min()) / 2).shift(26)
    
    # CCI (Manual MAD)
    sma_tp = tp.rolling(20).mean()
    def calc_mad(x): return np.mean(np.abs(x - np.mean(x)))
    mad = tp.rolling(20).apply(calc_mad, raw=True)
    h['CCI'] = (tp - sma_tp) / (0.015 * mad)
    
    # WR
    hh = h['High'].rolling(14).max(); ll = h['Low'].rolling(14).min()
    h['WR'] = -100 * (hh - h['Close']) / (hh - ll)
    
    # CMF
    mfm = ((h['Close'] - h['Low']) - (h['High'] - h['Close'])) / (h['High'] - h['Low'])
    mfv = mfm * h['Volume']
    h['CMF'] = mfv.rolling(20).sum() / h['Volume'].rolling(20).sum()
    
    # MACD/RSI/BOLL
    exp12 = h['Close'].ewm(span=12).mean(); exp26 = h['Close'].ewm(span=26).mean()
    h['MACD'] = exp12 - exp26; h['Signal'] = h['MACD'].ewm(span=9).mean(); h['Hist'] = h['MACD'] - h['Signal']
    delta = h['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss; h['RSI'] = 100 - (100 / (1 + rs))
    h['UPPER'] = h['MA20'] + 2*h['STD20']; h['LOWER'] = h['MA20'] - 2*h['STD20']
    
    # KDJ
    low_min = h['Low'].rolling(9).min(); high_max = h['High'].rolling(9).max()
    h['RSV'] = (h['Close'] - low_min) / (high_max - low_min) * 100
    h['K'] = h['RSV'].ewm(com=2).mean(); h['D'] = h['K'].ewm(com=2).mean(); h['J'] = 3 * h['K'] - 2 * h['D']
    h['OBV'] = (np.sign(h['Close'].diff()) * h['Volume']).fillna(0).cumsum()
    
    # Donchian
    h['DC_Upper'] = h['High'].rolling(20).max(); h['DC_Lower'] = h['Low'].rolling(20).min()

    safe_info = s.info if s.info is not None else {}
    
    # Earnings Date
    earnings_date = "N/A"
    try:
        cal = s.calendar
        if isinstance(cal, dict) and cal:
            if 'Earnings Date' in cal: earnings_date = str(cal['Earnings Date'][0])
            elif 'Earnings High' in cal: earnings_date = str(cal.get('Earnings Date', [])[0])
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            earnings_date = cal.iloc[0, 0].strftime("%Y-%m-%d")
    except: pass

    return {
        "history": h, "info": safe_info, "earnings_date": earnings_date,
        "error": None
    }

# ================= 3. 辅助函数 =================
def fmt_pct(v): return f"{v:.2%}" if isinstance(v, (int, float)) else "-"
def fmt_num(v): return f"{v:.2f}" if isinstance(v, (int, float)) else "-"
def fmt_big(v): 
    if not isinstance(v, (int, float)): return "-"
    if v > 1e12: return f"{v/1e12:.2f}T"
    if v > 1e9: return f"{v/1e9:.2f}B"
    if v > 1e6: return f"{v/1e6:.2f}M"
    return str(v)
def smart_translate(t, d): 
    if not isinstance(t, str): return t
    for k,v in d.items(): 
        if k.lower() in t.lower(): return v
    return t

# ================= 4. 业务逻辑 =================

def process_news(news_list):
    # [FIX V90] 内部导入，防止 NameError
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

def calculate_volume_profile(df, bins=50):
    price_min = df['Low'].min()
    price_max = df['High'].max()
    hist = np.histogram(df['Close'], bins=bins, range=(price_min, price_max), weights=df['Volume'])
    return hist[1][:-1], hist[0]

def calculate_seasonality(df):
    if df.empty: return None
    df = df.copy()
    df['Month'] = df.index.month
    df['Ret'] = df['Close'].pct_change()
    monthly_stats = df.groupby('Month')['Ret'].agg(['mean', lambda x: (x>0).sum() / len(x)])
    monthly_stats.columns = ['Avg Return', 'Win Rate']
    return monthly_stats

# ================= 5. 主程序 =================

if 'watchlist' not in st.session_state: st.session_state.watchlist = ['TSLA', 'NVDA', 'AAPL', 'AMD', 'PLTR']
if 'current_ticker' not in st.session_state: st.session_state.current_ticker = 'TSLA'

# 侧边栏
with st.sidebar:
    st.title("🦁 摩根·V1")
    new_ticker = st.text_input("🔍 搜索代码", "").upper()
    if new_ticker:
        st.session_state.current_ticker = new_ticker
        st.rerun()

    st.caption("我的自选 (实时)")
    for t in st.session_state.watchlist:
        p_data = fetch_realtime_price(t)
        chg = (p_data['price'] - p_data['prev']) / p_data['prev'] if p_data['prev'] else 0
        c_color = "#4ade80" if chg >= 0 else "#f87171"
        c1, c2 = st.columns([2, 1])
        if c1.button(f"{t}", key=f"btn_{t}"):
            st.session_state.current_ticker = t
            st.rerun()
        c2.markdown(f"<span style='color:{c_color}'>{chg:.2%}</span>", unsafe_allow_html=True)

# 页面路由
page = st.sidebar.radio("📌 导航", ["🚀 股票分析", "📖 功能说明书"])

if page == "🚀 股票分析":
    ticker = st.session_state.current_ticker
    
    # 1. 极速报价区域
    price_data = fetch_realtime_price(ticker)
    p = price_data['price']
    prev = price_data['prev']
    chg_val = p - prev
    chg_pct = chg_val / prev if prev else 0
    color = "#4ade80" if chg_val >= 0 else "#f87171"
    bg_color = "rgba(74, 222, 128, 0.1)" if chg_val >= 0 else "rgba(248, 113, 113, 0.1)"

    st.markdown(f"""
    <div class='price-container'>
        <div style='color:#9CA3AF; font-size:14px; font-weight:bold; letter-spacing:1px;'>{ticker} 实时报价</div>
        <div class='big-price' style='color:{color}'>${p:.2f}</div>
        <div class='price-change' style='background-color:{bg_color}; color:{color}'>
            {chg_val:+.2f} ({chg_pct:+.2%})
        </div>
        {f"<div class='ext-price'>🌙 {price_data['ext_label']}: ${price_data['ext_price']:.2f}</div>" if price_data['ext_price'] else ""}
    </div>
    """, unsafe_allow_html=True)

    c_btn = st.columns(4)
    c_btn[0].link_button("🔥 谷歌", f"https://www.google.com/search?q=why+is+{ticker}+moving")
    c_btn[1].link_button("🎯 目标价", f"https://www.google.com/search?q={ticker}+stock+target")
    c_btn[2].link_button("👽 Reddit", f"https://www.reddit.com/search/?q=${ticker}")
    c_btn[3].link_button("🐦 Twitter", f"https://twitter.com/search?q=${ticker}")

    # 2. 深度数据加载
    with st.spinner("🦁 正在调取机构底仓数据..."):
        heavy = fetch_heavy_data(ticker)

    if heavy['error']:
        st.warning(f"深度数据暂时不可用 (Rate Limit): {heavy['error']}")
    else:
        h = heavy['history']
        i = heavy['info']
        
        # 渲染 L-Box
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

            # 主图
            with st.expander("📈 机构趋势图 (SuperTrend + FVG)", expanded=True):
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'], name='K线'))
                fig.add_trace(go.Scatter(x=h.index, y=h['ST_Lower'], mode='markers', marker=dict(color='orange', size=2), name='止损线'))
                fig.add_trace(go.Scatter(x=h.index, y=h['VWAP'], line=dict(color='#fcd34d', width=1), name='VWAP'))
                fig.add_trace(go.Scatter(x=h.index, y=h['HMA'], line=dict(color='#ec4899', width=1), name='HMA'))
                for idx in range(len(h)-50, len(h)): 
                    if h['FVG_Bull'].iloc[idx]:
                        fig.add_shape(type="rect", x0=h.index[idx-2], y0=h['Low'].iloc[idx], x1=h.index[idx], y1=h['High'].iloc[idx-2], fillcolor="rgba(139, 92, 246, 0.3)", line_width=0)
                fig.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # 蒙特卡洛
            with st.expander("📅 历史季节性 & 蒙特卡洛预测"):
                c_seas, c_mc = st.columns(2)
                with c_seas:
                    seas = calculate_seasonality(h)
                    if seas is not None:
                        fig_seas = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_seas.add_trace(go.Bar(x=seas.index, y=seas['Avg Return']*100, name='平均回报', marker_color='#3b82f6'))
                        fig_seas.add_trace(go.Scatter(x=seas.index, y=seas['Win Rate']*100, name='胜率', line=dict(color='#f97316')), secondary_y=True)
                        fig_seas.update_layout(title="季节性", height=300, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_seas, use_container_width=True)
                with c_mc:
                    # 简化版蒙特卡洛
                    last_price = h['Close'].iloc[-1]; daily_vol = h['Close'].pct_change().std()
                    simulations = 50; days = 30; sim_df = pd.DataFrame() # 减少模拟次数防卡
                    for x in range(simulations):
                        price_series = [last_price]
                        for y in range(days): price_series.append(price_series[-1] * (1 + np.random.normal(0, daily_vol)))
                        sim_df[x] = price_series
                    fig_mc = go.Figure()
                    for col in sim_df.columns: fig_mc.add_trace(go.Scatter(y=sim_df[col], mode='lines', line=dict(color='rgba(59, 130, 246, 0.1)', width=1), showlegend=False))
                    fig_mc.update_layout(title="未来30天模拟", height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_mc, use_container_width=True)

            # 进阶指标
            with st.expander("📉 进阶指标 (Z-Score/ADX/CCI)"):
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

        # 核心数据
        st.subheader("📊 核心数据")
        c1, c2, c3 = st.columns(3)
        c1.metric("市值", fmt_big(i.get('marketCap')))
        c2.metric("做空比", fmt_pct(i.get('shortPercentOfFloat')))
        c3.metric("下次财报", heavy.get('earnings_date', 'N/A'))
        
        # 标签页 (资讯/持仓/估值/研报)
        tabs = st.tabs(["📰 资讯", "👥 持仓", "💰 估值", "🎓 深度研报"])
        
        with tabs[0]:
            news_df = process_news(heavy.get('news', [])) # [FIX] Safe get
            if not news_df.empty: st.dataframe(news_df[['时间','标题','价格','链接']], column_config={"链接": st.column_config.LinkColumn("阅读"), "价格": st.column_config.TextColumn("🎯", width="small")}, hide_index=True)
            else: st.info("暂无新闻")
            
        with tabs[1]:
            st.subheader("🏦 机构持仓")
            if heavy.get('inst') is not None: st.dataframe(heavy['inst'], use_container_width=True)
            st.subheader("🕴️ 内部交易")
            if heavy.get('insider') is not None: st.dataframe(heavy['insider'], use_container_width=True)
            
        with tabs[2]:
            st.subheader("💰 DCF 模型")
            safe_i = i if isinstance(i, dict) else {}
            peg = safe_i.get('pegRatio')
            if peg: st.caption(f"PEG: {peg} {'✅' if peg < 1 else '⚠️'}")
            g = st.slider("预期增长率 %", 0, 50, 15)
            eps = safe_i.get('trailingEps', 0)
            if eps > 0:
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

else:
    render_documentation()