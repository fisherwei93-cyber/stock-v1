import streamlit as st
import os
import datetime
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ================= 1. 铁律配置 (V87.1: 修复数据格式崩溃) =================
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if key in os.environ: del os.environ[key]

# 注入自定义图标 (Apple Touch Icon)
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
        
        /* 组件样式 */
        div[data-testid="stMetricValue"] {{ color: #fff !important; }}
        .streamlit-expanderHeader {{ background-color: #222 !important; color: #fff !important; border: 1px solid #444; }}
        .l-box {{ background-color: #FF9F1C; color: #000 !important; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .l-title {{ font-size: 18px; font-weight: 900; border-bottom: 2px solid #000; padding-bottom: 8px; margin-bottom: 12px; color: #000; }}
        .l-item {{ display: flex; justify-content: space-between; border-bottom: 1px dashed rgba(0,0,0,0.2); padding: 4px 0; color: #000; font-weight: 600; }}
        .wl-row {{ background: #1A1A1A; padding: 12px; margin-bottom: 8px; border-radius: 6px; border-left: 4px solid #555; display: flex; justify-content: space-between; align-items: center; color: #fff; }}
        .social-box {{ display: flex; gap: 10px; margin-top: 10px; }}
    </style>
""", unsafe_allow_html=True)

import yfinance as yf

# ================= 2. 智能数据引擎 =================

# 🔴 快通道：实时价格 (15s缓存)
@st.cache_data(ttl=15, show_spinner=False)
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
        
        ext_price, ext_label = None, ""
        try:
            info = s.info if s.info is not None else {}
            pm_price = info.get('preMarketPrice')
            post_price = info.get('postMarketPrice')
            if pm_price and abs(pm_price - price) > 0.01:
                ext_price, ext_label = pm_price, "盘前"
            elif post_price and abs(post_price - price) > 0.01:
                ext_price, ext_label = post_price, "盘后"
        except: pass

        return {"price": price, "prev": prev, "ext_price": ext_price, "ext_label": ext_label}
    except:
        return {"price": 0, "prev": 0, "ext_price": None, "ext_label": ""}

# 🔵 慢通道：深度数据 (1h缓存)
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_heavy_data(ticker):
    try:
        s = yf.Ticker(ticker)
        h = pd.DataFrame()
        # 重试机制
        for _ in range(3):
            try:
                h = s.history(period="2y")
                if not h.empty: break
                time.sleep(1)
            except: pass
        
        if h.empty: return {"history": pd.DataFrame(), "info": {}, "error": "No Data"}

        # 指标计算
        h['MA20'] = h['Close'].rolling(20).mean()
        h['MA200'] = h['Close'].rolling(200).mean()
        
        # SuperTrend (ATR Based)
        h['TR'] = np.maximum(h['High'] - h['Low'], np.abs(h['High'] - h['Close'].shift(1)))
        h['ATR'] = h['TR'].rolling(10).mean()
        h['ST_Lower'] = ((h['High']+h['Low'])/2) - (3 * h['ATR'])
        
        # VWAP
        v = h['Volume'].values
        tp = (h['High'] + h['Low'] + h['Close']) / 3
        h['VWAP'] = (tp * v).cumsum() / v.cumsum()

        # FVG
        h['FVG_Bull'] = (h['Low'] > h['High'].shift(2))
        
        safe_info = s.info if s.info is not None else {}
        
        # [FIX] 修复 'dict' object has no attribute 'empty'
        earnings_date = "N/A"
        try:
            cal = s.calendar
            # 情况1: 是字典 (New yfinance)
            if isinstance(cal, dict) and cal:
                # 尝试获取常见的键值
                if 'Earnings Date' in cal:
                    val = cal['Earnings Date']
                    earnings_date = str(val[0]) if len(val) > 0 else "N/A"
                elif 'Earnings High' in cal: # 有时候key变了
                    val = cal.get('Earnings Date', [])
                    earnings_date = str(val[0]) if len(val) > 0 else "N/A"
            # 情况2: 是表格 (Old yfinance)
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                earnings_date = cal.iloc[0, 0].strftime("%Y-%m-%d")
        except:
            earnings_date = "N/A"

        return {
            "history": h, "info": safe_info, "earnings_date": earnings_date,
            "error": None
        }
    except Exception as e:
        return {"history": pd.DataFrame(), "info": {}, "error": str(e)}

def fmt_big(v):
    if not isinstance(v, (int, float)): return "-"
    if v > 1e12: return f"{v/1e12:.2f}T"
    if v > 1e9: return f"{v/1e9:.2f}B"
    return str(v)

# ================= 3. 主程序 =================

if 'watchlist' not in st.session_state: st.session_state.watchlist = ['TSLA', 'NVDA', 'AAPL', 'AMD', 'PLTR']
if 'current_ticker' not in st.session_state: st.session_state.current_ticker = 'TSLA'

with st.sidebar:
    st.title("🦁 摩根·V1")
    new_ticker = st.text_input("🔍 搜索 (如 AAPL)", "").upper()
    if new_ticker:
        st.session_state.current_ticker = new_ticker
        st.rerun()

    st.caption("我的自选")
    for t in st.session_state.watchlist:
        p_data = fetch_realtime_price(t)
        chg = (p_data['price'] - p_data['prev']) / p_data['prev'] if p_data['prev'] else 0
        c_color = "#4ade80" if chg >= 0 else "#f87171"
        c1, c2 = st.columns([2, 1])
        if c1.button(f"{t}", key=f"btn_{t}"):
            st.session_state.current_ticker = t
            st.rerun()
        c2.markdown(f"<span style='color:{c_color}'>{chg:.2%}</span>", unsafe_allow_html=True)

ticker = st.session_state.current_ticker

# 1. 极速报价
price_data = fetch_realtime_price(ticker)
p = price_data['price']
prev = price_data['prev']
chg_val = p - prev
chg_pct = chg_val / prev if prev else 0
color = "#4ade80" if chg_val >= 0 else "#f87171"
bg_color = "rgba(74, 222, 128, 0.1)" if chg_val >= 0 else "rgba(248, 113, 113, 0.1)"

st.markdown(f"""
<div class='price-container'>
    <div style='color:#9CA3AF; font-size:14px; font-weight:bold; letter-spacing:1px;'>{ticker} 实时</div>
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

# 2. 深度数据
with st.spinner("🦁 正在调取机构底仓数据..."):
    heavy = fetch_heavy_data(ticker)

if heavy['error']:
    st.warning(f"深度数据暂时不可用: {heavy['error']}")
else:
    h = heavy['history']
    i = heavy['info']
    
    if not h.empty:
        curr = h['Close'].iloc[-1]
        ma20 = h['MA20'].iloc[-1]
        ma200 = h['MA200'].iloc[-1]
        res = h['High'].tail(20).max()
        sup = h['Low'].tail(20).min()
        
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
            for idx in range(len(h)-50, len(h)): 
                if h['FVG_Bull'].iloc[idx]:
                    fig.add_shape(type="rect", x0=h.index[idx-2], y0=h['Low'].iloc[idx], x1=h.index[idx], y1=h['High'].iloc[idx-2], fillcolor="rgba(139, 92, 246, 0.3)", line_width=0)
            fig.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 核心数据")
    c1, c2, c3 = st.columns(3)
    c1.metric("市值", fmt_big(i.get('marketCap')))
    c2.metric("做空比", fmt_pct(i.get('shortPercentOfFloat')))
    c3.metric("下次财报", heavy.get('earnings_date', 'N/A'))
    
    with st.expander("📖 快速功能指南"):
        st.markdown("""
        * **价格刷新**：每 15 秒自动更新，不消耗流量。
        * **紫色方块**：机构缺口 (FVG)，通常是支撑位。
        * **黄线 (VWAP)**：机构成本线，线上看多，线下看空。
        """)