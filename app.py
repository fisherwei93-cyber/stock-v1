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
import yfinance as yf

# ================= 1. 铁律配置 =================
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if key in os.environ:
        del os.environ[key]

ICON_URL = "https://cdn-icons-png.flaticon.com/512/10452/10452449.png"

st.set_page_config(page_title="摩根·V1 (Full)", layout="wide", page_icon="🦁")

# ================= 2. 样式死锁 =================
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

    /* 核心黄框 (Vision L-Box) - 完美复刻图1 */
    .l-box {{
        background-color: #FF9F1C; color: #000000 !important; padding: 15px;
        border-radius: 8px; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(255, 159, 28, 0.4);
    }}
    .l-title {{ font-size: 18px; font-weight: 900; border-bottom: 2px solid #000; padding-bottom: 8px; margin-bottom: 10px; color: #000; }}
    .l-sub {{ font-size: 14px; font-weight: 800; margin-top: 8px; margin-bottom: 4px; color: #333; text-transform: uppercase; }}
    .l-item {{ display: flex; justify-content: space-between; align-items: center; font-size: 14px; font-weight: 600; border-bottom: 1px dashed rgba(0,0,0,0.2); padding: 3px 0; color: #000; }}
    
    /* 标签与按钮 */
    .tg-s {{ background: rgba(0,0,0,0.1); padding: 1px 5px; border-radius: 4px; font-size: 11px; margin-left: 6px; color: #333; }}
    .earning-row {{ display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #333; font-size: 13px; }}
    .earning-soon {{ border-left: 3px solid #ef4444; background: rgba(239, 68, 68, 0.1); }}
    
    /* 多空博弈 */
    .thesis-col {{ flex: 1; padding: 10px; border-radius: 6px; font-size: 13px; margin-top:5px; }}
    .thesis-bull {{ background: rgba(6, 78, 59, 0.8); border: 1px solid #34d399; color: #fff; }}
    .thesis-bear {{ background: rgba(127, 29, 29, 0.8); border: 1px solid #f87171; color: #fff; }}
</style>
""", unsafe_allow_html=True)

# ================= 3. 数据引擎 (全功能+反封锁) =================

def fmt_pct(v): return f"{v:.2%}" if isinstance(v, (int, float)) else "-"
def mk_range(v): return f"{v*0.985:.1f}-{v*1.015:.1f}" if isinstance(v, (int, float)) else "-"

@st.cache_data(ttl=30, show_spinner=False)
def fetch_realtime_price(ticker):
    try:
        s = yf.Ticker(ticker)
        try: price = s.fast_info.last_price; prev = s.fast_info.previous_close
        except: 
            info = s.info if s.info else {}
            price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            prev = info.get('previousClose', price)
        
        ext_price, ext_label = None, ""
        try:
            info = s.info if s.info else {}
            pm = info.get('preMarketPrice'); post = info.get('postMarketPrice')
            if pm and abs(pm - price) > 0.01: ext_price, ext_label = pm, "盘前"
            elif post and abs(post - price) > 0.01: ext_price, ext_label = post, "盘后"
        except: pass
        return {"price": price, "prev": prev, "ext_price": ext_price, "ext_label": ext_label}
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

    # --- 指标计算 (为了 L-Box 和 Thesis) ---
    h['MA20'] = h['Close'].rolling(20).mean()
    h['MA60'] = h['Close'].rolling(60).mean()
    h['MA120'] = h['Close'].rolling(120).mean()
    h['MA200'] = h['Close'].rolling(200).mean()
    
    # SuperTrend
    h['TR'] = np.maximum(h['High'] - h['Low'], np.abs(h['High'] - h['Close'].shift(1)))
    h['ATR'] = h['TR'].rolling(10).mean()
    h['ST_Lower'] = ((h['High']+h['Low'])/2) - (3 * h['ATR'])
    
    # VWAP & FVG
    v = h['Volume'].values; tp = (h['High'] + h['Low'] + h['Close']) / 3
    h['VWAP'] = (tp * v).cumsum() / v.cumsum()
    h['FVG_Bull'] = (h['Low'] > h['High'].shift(2))
    
    # MACD & RSI (用于 L-Box 诊断)
    exp12 = h['Close'].ewm(span=12).mean(); exp26 = h['Close'].ewm(span=26).mean()
    h['MACD'] = exp12 - exp26
    delta = h['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss; h['RSI'] = 100 - (100 / (1 + rs))

    # [Comparison Data]
    cmp_norm = pd.DataFrame()
    try:
        h_recent = h.iloc[-504:] 
        spy = yf.Ticker("SPY").history(period="2y")['Close']
        qqq = yf.Ticker("QQQ").history(period="2y")['Close']
        idx = h_recent.index.intersection(spy.index).intersection(qqq.index)
        cmp_df = pd.DataFrame({
            ticker: h_recent.loc[idx, 'Close'], "SP500": spy.loc[idx], "Nasdaq": qqq.loc[idx]
        })
        start = -252 if len(cmp_df)>252 else 0
        cmp_norm = cmp_df.iloc[start:] / cmp_df.iloc[start] - 1
    except: pass

    safe_info = s.info if s.info is not None else {}
    return {"history": h, "info": safe_info, "compare": cmp_norm, "error": None}

@st.cache_data(ttl=43200, show_spinner=False)
def fetch_sector_earnings():
    sectors = {
        "💻 科技": ["NVDA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA"],
        "🏦 金融": ["JPM", "BAC", "V"],
        "💎 芯片": ["AMD", "AVGO", "TSM"]
    }
    flat_list = []
    for sec, tickers in sectors.items():
        for t in tickers: flat_list.append((t, sec))
    results = []
    today = datetime.date.today()
    for t, sec in flat_list:
        try:
            s = yf.Ticker(t); cal = s.calendar; e_date = None
            if isinstance(cal, dict) and cal:
                if 'Earnings Date' in cal: e_date = cal['Earnings Date'][0]
            elif isinstance(cal, pd.DataFrame) and not cal.empty: e_date = cal.iloc[0, 0]
            if e_date:
                ed = datetime.datetime.strptime(str(e_date).split()[0], "%Y-%m-%d").date()
                if ed >= today: results.append({"Code": t, "Date": str(ed), "Days": (ed - today).days})
        except: pass
    return sorted(results, key=lambda x: x['Days']) if results else []

@st.cache_data(ttl=3600)
def fetch_correlation_data(ticker):
    try:
        benchmarks = ['SPY', 'QQQ', 'GLD', 'BTC-USD']
        data = yf.download([ticker] + benchmarks, period="1y", progress=False)['Close']
        if data.empty: return None
        data = data.pct_change().dropna()
        if isinstance(data.columns, pd.MultiIndex): data.columns = [c[0] for c in data.columns]
        return data.corrwith(data[ticker]).drop(ticker)
    except: return None

# ================= 4. 逻辑核心 (还原 L-Box 与 Thesis) =================

def calculate_vision_analysis(df, info):
    # [RESTORED] 复活的 L-Box 详细逻辑
    if len(df) < 250: return None
    curr = df['Close'].iloc[-1]
    
    ma20 = df['MA20'].iloc[-1]; ma60 = df['MA60'].iloc[-1]; ma200 = df['MA200'].iloc[-1]
    low_52w = df['Low'].tail(250).min(); high_52w = df['High'].tail(250).max()
    
    pts = []
    # 压力位逻辑
    if curr < ma20: pts.append({"t":"res", "l":"小", "v":ma20, "d":"MA20/反压"})
    if curr < ma60: pts.append({"t":"res", "l":"中", "v":ma60, "d":"MA60/趋势"})
    if curr < high_52w: pts.append({"t":"res", "l":"强", "v":high_52w, "d":"前高"})
    
    # 支撑位逻辑
    if curr > ma20: pts.append({"t":"sup", "l":"小", "v":ma20, "d":"MA20/月线"})
    if curr > ma60: pts.append({"t":"sup", "l":"中", "v":ma60, "d":"MA60/趋势"})
    if curr > ma200: pts.append({"t":"sup", "l":"强", "v":ma200, "d":"MA200/牛熊"})
    
    def filter_pts(p_list, reverse=False):
        p_list = sorted(p_list, key=lambda x:x['v'], reverse=reverse)
        return p_list[:2] # 取最近的2个

    sups = filter_pts([p for p in pts if p['t']=="sup"], reverse=True)
    ress = filter_pts([p for p in pts if p['t']=="res"], reverse=False)
    
    # 估值与技术
    if not isinstance(info, dict): info = {}
    eps_fwd = info.get('forwardEps'); val_data = f"{eps_fwd*25:.0f}-{eps_fwd*35:.0f}" if eps_fwd else "N/A"
    
    rsi = df['RSI'].iloc[-1]; macd_val = df['MACD'].iloc[-1]
    tech_str = f"RSI({rsi:.0f}) | {'MACD金叉' if macd_val>0 else 'MACD死叉'}"
    
    return {"growth": info.get('revenueGrowth', 0), "val_range": val_data, "sups": sups, "ress": ress, "tech": tech_str}

def generate_bull_bear_thesis(df, info):
    # [RESTORED] 复活的多空博弈
    if df.empty: return [], []
    bulls = []; bears = []
    curr = df['Close'].iloc[-1]; ma200 = df['MA200'].iloc[-1]; rsi = df['RSI'].iloc[-1]
    
    if curr > ma200: bulls.append("股价站上年线 (长期趋势向上)")
    else: bears.append("股价跌破年线 (长期趋势向下)")
    
    if rsi < 30: bulls.append("RSI超卖 (存在反弹需求)")
    if rsi > 70: bears.append("RSI超买 (存在回调风险)")
    
    short = info.get('shortPercentOfFloat', 0)
    if short and short > 0.2: bulls.append("高做空比 (逼空潜力)")
    
    rev_g = info.get('revenueGrowth', 0)
    if rev_g > 0.2: bulls.append("高成长 (营收增速 > 20%)")
    
    while len(bulls) < 3: bulls.append("暂无更多明显信号")
    while len(bears) < 3: bears.append("暂无更多明显信号")
    return bulls[:3], bears[:3]

# ================= 5. 主程序 =================
if 'watchlist' not in st.session_state: st.session_state.watchlist = ['TSLA', 'NVDA', 'AAPL', 'AMD', 'PLTR']
if 'current_ticker' not in st.session_state: st.session_state.current_ticker = 'TSLA'

# Sidebar
with st.sidebar:
    st.title("🦁 摩根·V1")
    
    with st.expander("📺 视频分析", expanded=True):
        yt_url = st.text_input("YouTube Link", placeholder="粘贴URL...")
        if st.button("🚀 提取"): st.info("功能保留")

    new_ticker = st.text_input("🔍 搜索", "").upper()
    if new_ticker: st.session_state.current_ticker = new_ticker; st.rerun()

    # 财报日历
    st.markdown("---")
    st.caption("📅 财报雷达")
    earnings_list = fetch_sector_earnings()
    if earnings_list:
        urgent = [x for x in earnings_list if x['Days'] <= 7]
        if urgent:
            for item in urgent:
                st.markdown(f"<div class='earning-row earning-soon'><span style='color:#fff;font-weight:bold'>{item['Code']}</span><span style='color:#cbd5e1'>{item['Date']} (T-{item['Days']})</span></div>", unsafe_allow_html=True)
        else: st.caption("近期无关注财报")

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

# Main Page
page = st.sidebar.radio("📌 导航", ["🚀 股票分析", "📖 功能说明书"])

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
        {f"<div class='ext-price'>🌙 {p_data['ext_label']}: ${p_data['ext_price']:.2f}</div>" if p_data['ext_price'] else ""}
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

    if not h.empty:
        # [RESTORED] L-Box (Detailed Version)
        l_an = calculate_vision_analysis(h, i)
        if l_an:
            res_rows = "".join([f"<div class='l-item'><span>压力 ({p['d']})</span><span style='color:#fdba74'>{mk_range(p['v'])}<span class='tg-s'>{p['l']}</span></span></div>" for p in l_an['ress']])
            sup_rows = "".join([f"<div class='l-item'><span>支撑 ({p['d']})</span><span style='color:#86efac'>{mk_range(p['v'])}<span class='tg-s'>{p['l']}</span></span></div>" for p in l_an['sups']])
            
            st.markdown(f"""
            <div class='l-box'>
                <div class='l-title'>🦁 视野·交易计划 ({ticker})</div>
                <div class='l-sub'>增速与估值</div>
                <div class='l-item'><span>未来增速 (Rev)</span><span>{fmt_pct(l_an['growth'])}</span></div>
                <div class='l-item'><span>前瞻合理估值 (25x-35x)</span><span style='font-weight:bold'>{l_an['val_range']}</span></div>
                <div class='l-item'><span>技术面诊断</span><span style='font-weight:bold; color:#2563EB'>{l_an['tech']}</span></div>
                <div class='l-sub'>关键点位 (Support/Resist)</div>
                {res_rows}
                {sup_rows}
            </div>
            """, unsafe_allow_html=True)

        # [RESTORED] Comparison Chart
        st.subheader("🆚 跑赢大盘了吗?")
        cmp = heavy.get('compare', pd.DataFrame())
        if not cmp.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=cmp.index, y=cmp[ticker]*100, name=ticker, line=dict(width=3, color='#3b82f6')))
            fig2.add_trace(go.Scatter(x=cmp.index, y=cmp['SP500']*100, name="SP500", line=dict(width=1.5, color='#9ca3af', dash='dot')))
            fig2.add_trace(go.Scatter(x=cmp.index, y=cmp['Nasdaq']*100, name="Nasdaq", line=dict(width=1.5, color='#f97316', dash='dot')))
            fig2.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

        # [RESTORED] Bull/Bear Thesis
        bulls, bears = generate_bull_bear_thesis(h, i)
        with st.expander("🐂 vs 🐻 智能多空博弈 (AI Thesis)", expanded=True):
            c_bull, c_bear = st.columns(2)
            with c_bull: st.markdown(f"<div class='thesis-col thesis-bull'><b>🚀 多头逻辑</b><br>{'<br>'.join([f'✅ {b}' for b in bulls])}</div>", unsafe_allow_html=True)
            with c_bear: st.markdown(f"<div class='thesis-col thesis-bear'><b>🔻 空头逻辑</b><br>{'<br>'.join([f'⚠️ {b}' for b in bears])}</div>", unsafe_allow_html=True)

        # Main Chart
        with st.expander("📈 机构趋势图 (SuperTrend)", expanded=True):
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'], name='K线'))
            fig.add_trace(go.Scatter(x=h.index, y=h['ST_Lower'], mode='markers', marker=dict(color='orange', size=2), name='止损线'))
            fig.add_trace(go.Scatter(x=h.index, y=h['VWAP'], line=dict(color='#fcd34d', width=1), name='VWAP'))
            for idx in range(len(h)-50, len(h)): 
                if h['FVG_Bull'].iloc[idx]: fig.add_shape(type="rect", x0=h.index[idx-2], y0=h['Low'].iloc[idx], x1=h.index[idx], y1=h['High'].iloc[idx-2], fillcolor="rgba(139, 92, 246, 0.3)", line_width=0)
            fig.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        # Seasonality & Monte Carlo
        with st.expander("📅 季节性 & 蒙特卡洛"):
            c_seas, c_mc = st.columns(2)
            with c_seas:
                seas = calculate_seasonality(h)
                if seas is not None:
                    fig_seas = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_seas.add_trace(go.Bar(x=seas.index, y=seas['Avg Return']*100, name='平均回报', marker_color='#3b82f6'))
                    fig_seas.add_trace(go.Scatter(x=seas.index, y=seas['Win Rate']*100, name='胜率', line=dict(color='#f97316')), secondary_y=True)
                    fig_seas.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_seas, use_container_width=True)
            with c_mc:
                last_price = h['Close'].iloc[-1]; daily_vol = h['Close'].pct_change().std()
                simulations = 50; days = 30; sim_df = pd.DataFrame()
                for x in range(simulations):
                    price_series = [last_price]
                    for y in range(days): price_series.append(price_series[-1] * (1 + np.random.normal(0, daily_vol)))
                    sim_df[x] = price_series
                fig_mc = go.Figure()
                for col in sim_df.columns: fig_mc.add_trace(go.Scatter(y=sim_df[col], mode='lines', line=dict(color='rgba(59, 130, 246, 0.1)', width=1), showlegend=False))
                fig_mc.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_mc, use_container_width=True)
                final_prices = sim_df.iloc[-1].values
                p5 = np.percentile(final_prices, 5); p95 = np.percentile(final_prices, 95)
                st.markdown(f"<div class='mc-box'><span style='color:#fca5a5'>📉 底线(P5): <b>${p5:.2f}</b></span> <span style='color:#86efac'>🚀 乐观(P95): <b>${p95:.2f}</b></span></div>", unsafe_allow_html=True)

    # Core Data
    st.subheader("📊 核心数据")
    c1, c2, c3 = st.columns(3)
    safe_i = i if isinstance(i, dict) else {}
    c1.metric("市值", fmt_big(safe_i.get('marketCap')))
    c2.metric("做空比", fmt_pct(safe_i.get('shortPercentOfFloat')))
    c3.metric("股息率", fmt_pct(safe_i.get('dividendYield')))

    # [RESTORED] Macro Correlation
    with st.expander("🌍 宏观联动 (BTC/Gold/SPY)", expanded=False):
        corrs = fetch_correlation_data(ticker)
        if corrs is not None: st.bar_chart(corrs)

    # Tabs
    tabs = st.tabs(["📰 资讯", "💰 估值", "🎓 深度研报"])

    with tabs[0]:
        news_df = process_news(heavy.get('news', []))
        if not news_df.empty: st.dataframe(news_df[['时间','标题','价格','链接']], column_config={"链接": st.column_config.LinkColumn("阅读"), "价格": st.column_config.TextColumn("🎯", width="small")}, hide_index=True)
        else: st.info("暂无新闻")
        
    with tabs[1]:
        st.subheader("⚖️ 格雷厄姆合理价")
        eps = safe_i.get('trailingEps', 0); bvps = safe_i.get('bookValue', 0)
        rt_price = p if p>0 else h['Close'].iloc[-1] if not h.empty else 0
        if eps and bvps and eps > 0 and bvps > 0 and rt_price > 0:
            graham = (22.5 * eps * bvps) ** 0.5
            upside = (graham - rt_price) / rt_price
            st.metric("Graham Number", f"${graham:.2f}", f"{upside:.1%} Upside")
        else: st.error("数据不足")
        
        st.subheader("💰 DCF 模型")
        peg = safe_i.get('pegRatio')
        if peg: st.caption(f"PEG: {peg} {'✅' if peg < 1 else '⚠️'}")
        g = st.slider("预期增长率 %", 0, 50, 15)
        if eps and eps > 0:
            val = (eps * ((1+g/100)**5) * 25) / (1.1**5)
            st.metric("估值", f"${val:.2f}")

    with tabs[2]:
        st.header(f"🎓 {ticker} 深度研报")
        st.markdown(f"<div class='report-text'>{safe_i.get('longBusinessSummary', '暂无描述')}</div>", unsafe_allow_html=True)
else:
    st.title("📚 摩根·功能说明书 (Wiki)")
    st.write("文档内容...")