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

st.set_page_config(page_title="摩根·V1 (Final)", layout="wide", page_icon="🦁")

# ================= 2. 样式死锁 (UI) =================
st.markdown(f"""
<head>
    <link rel="apple-touch-icon" href="{ICON_URL}">
    <link rel="icon" type="image/png" href="{ICON_URL}">
</head>
<style>
    /* 全局背景 */
    .stApp {{ background-color: #000000 !important; color: #FFFFFF !important; }}
    section[data-testid="stSidebar"] {{ background-color: #111111 !important; }}
    header {{ visibility: visible !important; }}

    /* 核心高亮 */
    div[data-testid="stMetricValue"] {{
        color: #FFFFFF !important; font-size: 28px !important; font-weight: 900 !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }}
    div[data-testid="stMetricLabel"] {{ color: #9CA3AF !important; font-weight: 700 !important; }}
    
    /* 侧边栏样式 */
    .earning-card {{ background: #1e1b4b; border-left: 4px solid #6366f1; padding: 8px; margin-bottom: 6px; border-radius: 4px; }}
    .earning-alert {{ background: #450a0a; border-left: 4px solid #ef4444; animation: pulse 2s infinite; }}
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }}
        70% {{ box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
    }}
    .ec-row {{ display: flex; justify-content: space-between; align-items: center; font-size: 13px; }}
    .ec-ticker {{ font-weight: bold; color: #fff; }}
    .ec-date {{ color: #cbd5e1; font-family: monospace; }}
    .ec-time {{ font-size: 11px; color: #fbbf24; margin-left: 5px; font-weight: bold; }}
    .ec-sector {{ font-size: 10px; padding: 1px 4px; border-radius: 3px; background: #333; color: #aaa; margin-top: 4px; display: inline-block;}}

    /* 核心报价盘 */
    .price-container {{ background: #1A1A1A; padding: 20px; border-radius: 15px; border: 1px solid #333; text-align: center; margin-bottom: 20px; }}
    .big-price {{ font-size: 56px !important; font-weight: 900 !important; color: #FFFFFF; line-height: 1.1; text-shadow: 0 0 20px rgba(255,255,255,0.1); }}
    .price-change {{ font-size: 24px !important; font-weight: bold; padding: 5px 15px; border-radius: 8px; display: inline-block; }}
    .ext-price {{ font-size: 16px !important; color: #9CA3AF; margin-top: 8px; font-family: monospace; }}

    /* 视野黄框 */
    .l-box {{ background-color: #FF9F1C; color: #000000 !important; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(255, 159, 28, 0.4); }}
    .l-title {{ font-size: 18px; font-weight: 900; border-bottom: 2px solid #000; padding-bottom: 8px; margin-bottom: 10px; color: #000; }}
    .l-sub {{ font-size: 14px; font-weight: 800; margin-top: 8px; margin-bottom: 4px; color: #333; text-transform: uppercase; }}
    .l-item {{ display: flex; justify-content: space-between; align-items: center; font-size: 14px; font-weight: 600; border-bottom: 1px dashed rgba(0,0,0,0.2); padding: 3px 0; color: #000; }}
    
    /* 历史搜索 */
    .hist-tag {{ display: inline-block; background: #333; color: #ccc; padding: 4px 10px; border-radius: 15px; font-size: 11px; margin: 3px; cursor: pointer; border: 1px solid #444; transition: 0.2s; }}
    .hist-tag:hover {{ border-color: #FF9F1C; color: #FF9F1C; background: #222; }}

    /* 持仓卡片 (Ultra Modern UI) */
    .hold-card {{ background: rgba(30, 30, 30, 0.6); border-bottom: 1px solid #333; padding: 10px; display: flex; justify-content: space-between; align-items: center; transition: 0.2s; cursor: pointer; }}
    .hold-card:hover {{ background: rgba(50, 50, 50, 0.8); border-color: #555; }}
    .hold-name {{ font-weight: 600; font-size: 13px; color: #f3f4f6; letter-spacing: 0.5px; text-decoration: none; }}
    .hold-sub {{ font-size: 11px; color: #9ca3af; margin-top: 2px; }}
    .hold-val {{ font-family: 'Segoe UI', monospace; font-weight: bold; color: #4ade80; font-size: 13px; }}
    .hold-bar-container {{ width: 60px; height: 4px; background: #333; border-radius: 2px; margin-top: 4px; margin-left: auto; }}
    .hold-bar-fill {{ height: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa); border-radius: 2px; }}
    .hold-link a {{ color: #fff; text-decoration: none; }}
    .hold-link a:hover {{ color: #FF9F1C; }}

    /* 指标解释框 */
    .ind-desc {{ background: #111; border-left: 3px solid #3b82f6; padding: 8px; margin-top: 5px; font-size: 12px; color: #ccc; }}

    /* 通用组件 */
    .tg-s {{ background: rgba(0,0,0,0.1); padding: 1px 5px; border-radius: 4px; font-size: 11px; margin-left: 6px; color: #333; }}
    .thesis-col {{ flex: 1; padding: 10px; border-radius: 6px; font-size: 13px; margin-top:5px; }}
    .thesis-bull {{ background: rgba(6, 78, 59, 0.8); border: 1px solid #34d399; color: #fff; }}
    .thesis-bear {{ background: rgba(127, 29, 29, 0.8); border: 1px solid #f87171; color: #fff; }}
    .score-card {{ background: #1A1A1A; padding: 15px; border-radius: 12px; text-align: center; border: 1px solid #333; margin-bottom: 15px; }}
    .sc-val {{ font-size: 42px; font-weight: 900; color: #4ade80; line-height: 1; }}
    .sc-lbl {{ font-size: 12px; color: #D1D5DB; font-weight: bold; }}
    .wl-row {{ background-color: #1A1A1A; padding: 12px; margin-bottom: 8px; border-radius: 6px; border-left: 4px solid #555; display: flex; justify-content: space-between; align-items: center; color: #FFFFFF; }}
    .social-box {{ display: flex; gap: 10px; margin-top: 10px; }}
    .mc-box {{ background: #0f172a; border: 1px solid #1e293b; padding: 10px; border-radius: 6px; margin-top:5px; }}
    .note-box {{ background: #1e1b4b; border-left: 4px solid #6366f1; padding: 10px; font-size: 12px; color: #e0e7ff; margin-top: 5px; border-radius: 4px; line-height: 1.6; }}
    .streamlit-expanderHeader {{ background-color: #222 !important; color: #fff !important; border: 1px solid #444; }}
    
    /* 研报样式 */
    .report-title {{ font-size: 22px; font-weight: 900; color: #FF9F1C; margin-bottom: 10px; border-left: 5px solid #FF9F1C; padding-left: 10px; }}
    .report-text {{ font-size: 15px; line-height: 1.8; color: #E5E7EB; margin-bottom: 20px; background: #1A1A1A; padding: 15px; border-radius: 8px; }}
    .guru-check {{ display: flex; align-items: center; margin-bottom: 8px; padding: 8px; background: #262626; border-radius: 6px; }}
    .wiki-card {{ background: #1A1A1A; border: 1px solid #333; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
    .wiki-title {{ font-size: 20px; font-weight: bold; color: #FF9F1C; margin-bottom: 15px; border-bottom: 1px solid #444; padding-bottom: 5px; }}
    .wiki-text {{ font-size: 14px; color: #E5E7EB; line-height: 1.8; margin-bottom: 10px; }}
    .wiki-tag {{ background: #374151; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 5px; border: 1px solid #555; }}
</style>
""", unsafe_allow_html=True)

# ================= 3. 数据引擎 =================

def fmt_pct(v): return f"{v:.2%}" if isinstance(v, (int, float)) else "-"
def fmt_num(v): return f"{v:.2f}" if isinstance(v, (int, float)) else "-"
def fmt_big(v): 
    if not isinstance(v, (int, float)): return "-"
    if v > 1e12: return f"{v/1e12:.2f}T"
    if v > 1e9: return f"{v/1e9:.2f}B"
    if v > 1e6: return f"{v/1e6:.2f}M"
    return str(v)
def mk_range(v): return f"{v*0.985:.1f}-{v*1.015:.1f}" if isinstance(v, (int, float)) else "-"
def smart_translate(t, d): 
    if not isinstance(t, str): return t
    for k,v in d.items(): 
        if k.lower() in t.lower(): return v
    return t

@st.cache_data(ttl=30, show_spinner=False)
def fetch_realtime_price(ticker):
    try:
        s = yf.Ticker(ticker)
        try: price = s.fast_info.last_price; prev = s.fast_info.previous_close
        except: 
            try: info = s.info if s.info else {}
            except: info = {}
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
    except: return {"price": 0, "prev": 0, "ext_price": None, "ext_label": ""}

# [FIX] V105.2: 模块级独立熔断，专门修复 NVDA 搜索崩溃问题
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financial_data_v105_2(ticker):
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

    # Core Indicators
    h['MA20'] = h['Close'].rolling(20).mean(); h['MA60'] = h['Close'].rolling(60).mean()
    h['MA120'] = h['Close'].rolling(120).mean(); h['MA200'] = h['Close'].rolling(200).mean()
    h['TR'] = np.maximum(h['High'] - h['Low'], np.abs(h['High'] - h['Close'].shift(1)))
    h['ATR'] = h['TR'].rolling(10).mean(); h['ST_Lower'] = ((h['High']+h['Low'])/2) - (3 * h['ATR'])
    v = h['Volume'].values; tp = (h['High'] + h['Low'] + h['Close']) / 3
    h['VWAP'] = (tp * v).cumsum() / v.cumsum(); h['FVG_Bull'] = (h['Low'] > h['High'].shift(2))
    h['STD20'] = h['Close'].rolling(20).std(); h['Z_Score'] = (h['Close'] - h['MA20']) / h['STD20']
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    period = 14
    wma_half = wma(h['Close'], period // 2); wma_full = wma(h['Close'], period)
    h['HMA'] = wma(2 * wma_half - wma_full, int(np.sqrt(period)))
    plus_dm = h['High'].diff(); minus_dm = h['Low'].diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm > 0] = 0; minus_dm = minus_dm.abs()
    tr14 = h['TR'].rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr14); minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di); h['ADX'] = dx.rolling(14).mean()
    sma_tp = tp.rolling(20).mean(); mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    h['CCI'] = (tp - sma_tp) / (0.015 * mad)
    h['MACD'] = h['Close'].ewm(span=12).mean() - h['Close'].ewm(span=26).mean()
    delta = h['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    h['RSI'] = 100 - (100 / (1 + (gain / loss)))

    cmp_norm = pd.DataFrame()
    try:
        h_recent = h.iloc[-504:] 
        spy = yf.Ticker("SPY").history(period="2y")['Close']; qqq = yf.Ticker("QQQ").history(period="2y")['Close']
        idx = h_recent.index.intersection(spy.index).intersection(qqq.index)
        cmp_df = pd.DataFrame({ticker: h_recent.loc[idx, 'Close'], "SP500": spy.loc[idx], "Nasdaq": qqq.loc[idx]})
        start = -252 if len(cmp_df)>252 else 0
        cmp_norm = cmp_df.iloc[start:] / cmp_df.iloc[start] - 1
    except: pass

    # [FIX] V105.2: 全模块独立熔断保护 (救活 NVDA)
    safe_info = {}; upgrades = None; inst = None; insider = None; fin = None
    try: safe_info = s.info if s.info else {}
    except: pass
    try: upgrades = s.upgrades_downgrades
    except: pass
    try: inst = s.institutional_holders
    except: pass
    try: insider = s.insider_transactions
    except: pass
    try: fin = s.quarterly_financials
    except: pass

    return {"history": h, "info": safe_info, "compare": cmp_norm, "error": None, "upgrades": upgrades, "inst": inst, "insider": insider, "fin": fin}

@st.cache_data(ttl=43200, show_spinner=False)
def fetch_sector_earnings():
    sectors = {"💻 科技": ["NVDA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA"], "🏦 金融": ["JPM", "BAC", "V", "COIN", "BLK"], "💊 医药": ["LLY", "JNJ", "PG", "KO", "MCD"], "⛽ 能源": ["XOM", "CVX", "CAT", "GE"], "💎 芯片": ["AMD", "AVGO", "TSM", "QCOM"]}
    results = []; today = datetime.date.today()
    for sec, tickers in sectors.items():
        for t in tickers:
            try:
                s = yf.Ticker(t); cal = s.calendar; e_date = None
                if isinstance(cal, dict) and cal: e_date = cal.get('Earnings Date', [None])[0]
                elif isinstance(cal, pd.DataFrame) and not cal.empty: e_date = cal.iloc[0, 0]
                if e_date:
                    ed = datetime.datetime.strptime(str(e_date).split()[0], "%Y-%m-%d").date()
                    if ed >= today:
                        # [NEW] Beijing Time Logic
                        time_label = "20:00 (盘前)" 
                        if t in ['NVDA', 'TSLA', 'AAPL', 'AMZN', 'GOOG', 'META', 'AMD', 'MSFT']: time_label = "次日04:20 (盘后)"
                        results.append({"Code": t, "Sector": sec, "Date": str(ed), "Days": (ed - today).days, "Time": time_label, "Sort": (ed - today).days})
            except: pass
    return sorted(results, key=lambda x: x['Sort']) if results else []

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

@st.cache_data(ttl=3600)
def fetch_mind_map_data(ticker):
    base = {"id": [ticker, "Competitors", "Supply", "Tech", "Macro"], "label": [f"🦁 {ticker}", "⚔️ 竞争对手", "⛓️ 供应链", "🧠 核心技术", "🌍 宏观"], "parent": ["", ticker, ticker, ticker, ticker], "color": ["#FF9F1C", "#ef4444", "#3b82f6", "#a855f7", "#10b981"]}
    custom = {"TSLA": {"Competitors": ["BYDDF", "RIVN", "LCID", "XPEV", "NIO"], "Supply": ["Panasonic", "CATL", "LG Energy", "Albemarle (Li)"], "Tech": ["FSD v12", "Dojo Supercomp", "4680 Cell", "Optimus Bot"], "Macro": ["Interest Rates", "EV Credits", "Oil Price"]}, "NVDA": {"Competitors": ["AMD", "INTC", "QCOM", "Google TPU"], "Supply": ["TSM (CoWoS)", "SK Hynix (HBM)", "Micron", "Samsung"], "Tech": ["CUDA Moat", "Hopper H100", "Blackwell B200", "Omniverse"], "Macro": ["AI Capex", "Crypto", "US-China Chips"]}, "AAPL": {"Competitors": ["MSFT", "GOOG", "Samsung", "Huawei"], "Supply": ["Foxconn", "TSMC", "Luxshare", "Qualcomm"], "Tech": ["Apple Silicon", "iOS Ecosystem", "Vision Pro", "Services"], "Macro": ["Consumer Spend", "Forex", "China Market"]}}
    data = custom.get(ticker, {"Competitors": ["Peer 1", "Peer 2"], "Supply": ["Supplier A", "Supplier B"], "Tech": ["Core Product", "R&D"], "Macro": ["Market Rate", "GDP"]})
    ids = base["id"][:]; labels = base["label"][:]; parents = base["parent"][:]
    for cat, items in data.items():
        for item in items: ids.append(f"{cat}-{item}"); labels.append(item); parents.append(cat)
    return ids, labels, parents

# ================= 4. 逻辑核心 =================

def calculate_vision_analysis(df, info):
    if len(df) < 250: return None
    if 'RSI' not in df.columns or 'MACD' not in df.columns: return None
    curr = df['Close'].iloc[-1]; ma20 = df['MA20'].iloc[-1]; ma60 = df['MA60'].iloc[-1]; ma200 = df['MA200'].iloc[-1]; high_52w = df['High'].tail(250).max()
    pts = []
    if curr < ma20: pts.append({"t":"res", "l":"小", "v":ma20, "d":"MA20/反压"})
    if curr < ma60: pts.append({"t":"res", "l":"中", "v":ma60, "d":"MA60/生命线"})
    if curr < high_52w: pts.append({"t":"res", "l":"强", "v":high_52w, "d":"52W前高"})
    if curr > ma20: pts.append({"t":"sup", "l":"小", "v":ma20, "d":"MA20/支撑"})
    if curr > ma60: pts.append({"t":"sup", "l":"中", "v":ma60, "d":"MA60/趋势"})
    if curr > ma200: pts.append({"t":"sup", "l":"强", "v":ma200, "d":"MA200/牛熊"})
    sups = sorted([p for p in pts if p['t']=="sup"], key=lambda x:x['v'], reverse=True)[:2]
    ress = sorted([p for p in pts if p['t']=="res"], key=lambda x:x['v'])[:2]
    eps_fwd = info.get('forwardEps'); val_data = f"{eps_fwd*25:.0f}-{eps_fwd*35:.0f}" if eps_fwd else "N/A"
    return {"growth": info.get('revenueGrowth', 0), "val_range": val_data, "sups": sups, "ress": ress, "tech": f"RSI({df['RSI'].iloc[-1]:.0f}) | {'MACD金叉' if df['MACD'].iloc[-1]>0 else 'MACD死叉'}"}

def generate_bull_bear_thesis(df, info):
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
    while len(bulls) < 3: bulls.append("暂无明显信号")
    while len(bears) < 3: bears.append("暂无明显信号")
    return bulls[:3], bears[:3]

def calculate_seasonality(df):
    if df.empty: return None
    df = df.copy(); df['Month'] = df.index.month; df['Ret'] = df['Close'].pct_change()
    monthly_stats = df.groupby('Month')['Ret'].agg(['mean', lambda x: (x>0).sum() / len(x)])
    monthly_stats.columns = ['Avg Return', 'Win Rate']
    return monthly_stats

def calculate_volume_profile(df, bins=50):
    price_min = df['Low'].min(); price_max = df['High'].max()
    hist = np.histogram(df['Close'], bins=bins, range=(price_min, price_max), weights=df['Volume'])
    return hist[1][:-1], hist[0]

def process_news(news_list):
    if not news_list: return pd.DataFrame()
    res = []; pat = r"(\$|USD)\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)|(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s?(美元|USD)"
    for n in news_list:
        title = n.get('title', 'No Title'); link = n.get('link', '#'); match = re.search(pat, title); price = "-"
        if match: vals = [g for g in match.groups() if g and g not in ['$','USD','美元']]; price = f"${vals[0]}" if vals else "-"
        ts = n.get('providerPublishTime', 0); t_str = pd.to_datetime(ts, unit='s').strftime('%m-%d %H:%M')
        res.append({"时间": t_str, "标题": title, "价格": price, "链接": link})
    return pd.DataFrame(res)

def calculate_quant_score(info, history):
    if not isinstance(info, dict): info = {}
    score = 50; notes = []
    if not history.empty:
        c = history['Close'].iloc[-1]; ma50 = history['Close'].rolling(50).mean().iloc[-1]
        if c > ma50: score += 15; notes.append("趋势向上")
    pe = info.get('forwardPE'); gr = info.get('revenueGrowth'); rec = info.get('recommendationMean')
    if pe and pe < 30: score += 10; notes.append("估值健康")
    if gr and gr > 0.15: score += 10; notes.append("高成长")
    if rec and rec < 2.0: score += 15; notes.append("机构强推")
    return min(100, max(0, int(score))), " | ".join(notes)

# ================= 5. 主程序 =================
if 'watchlist' not in st.session_state: st.session_state.watchlist = ['TSLA', 'NVDA', 'AAPL', 'AMD', 'PLTR']
if 'current_ticker' not in st.session_state: st.session_state.current_ticker = 'TSLA'
if 'search_history' not in st.session_state: st.session_state.search_history = []

with st.sidebar:
    st.title("🦁 摩根·V1")
    page = st.radio("📌 导航", ["🚀 股票分析", "🗓️ 财报地图", "📖 功能说明书"])
    if 'quant_score' in st.session_state:
        s, n = st.session_state.quant_score
        c = "#4ade80" if s>=60 else "#f87171"
        st.markdown(f"<div class='score-card'><div class='sc-lbl'>MORGAN SCORE</div><div class='sc-val' style='color:{c}'>{s}</div><div class='sc-lbl' style='color:#9CA3AF'>{n}</div></div>", unsafe_allow_html=True)
    with st.form(key='search_form'):
        c_s1, c_s2 = st.columns([3, 1])
        new_ticker = c_s1.text_input("搜索代码", label_visibility="collapsed").upper()
        if c_s2.form_submit_button("🔍") and new_ticker:
            st.session_state.current_ticker = new_ticker
            if new_ticker not in st.session_state.search_history: st.session_state.search_history.insert(0, new_ticker)
            st.session_state.search_history = st.session_state.search_history[:5]; st.rerun()
    if st.session_state.search_history:
        st.caption("📜 历史搜索")
        cols = st.columns(len(st.session_state.search_history))
        for idx, h_t in enumerate(st.session_state.search_history):
            with cols[idx]:
                if st.button(h_t, key=f"hist_{h_t}"): st.session_state.current_ticker = h_t; st.rerun()
    st.markdown("---"); st.caption("📅 财报雷达 (7天内高亮)")
    earnings_list = fetch_sector_earnings()
    if earnings_list:
        for item in earnings_list[:10]:
            is_urgent = item['Days'] <= 7; bg_style = "earning-alert" if is_urgent else "earning-card"; icon = "🚨" if is_urgent else "📅"
            st.markdown(f"<div class='earning-card {bg_style}'><div class='ec-row'><span class='ec-ticker'>{icon} {item['Code']}</span><span class='ec-date'>{item['Date']}</span></div><div class='ec-row'><span class='ec-sector'>{item['Sector']}</span><span class='ec-time'>{item['Time']}</span></div></div>", unsafe_allow_html=True)
    else: st.caption("数据更新中...")
    st.markdown("---"); st.caption("我的自选")
    for t in st.session_state.watchlist:
        p_d = fetch_realtime_price(t); chg = (p_d['price'] - p_d['prev']) / p_d['prev'] if p_d['prev'] else 0
        c_color = "#4ade80" if chg >= 0 else "#f87171"; c1, c2 = st.columns([2, 1])
        if c1.button(f"{t}", key=f"btn_{t}"): st.session_state.current_ticker = t; st.rerun()
        c2.markdown(f"<span style='color:{c_color}'>{chg:.2%}</span>", unsafe_allow_html=True)

if page == "🚀 股票分析":
    ticker = st.session_state.current_ticker
    p_data = fetch_realtime_price(ticker); p = p_data['price']; prev = p_data['prev']
    color = "#4ade80" if (p-prev) >= 0 else "#f87171"
    st.markdown(f"<div class='price-container'><div style='color:#9CA3AF; font-size:14px; font-weight:bold;'>{ticker} 实时报价</div><div class='big-price' style='color:{color}'>${p:.2f}</div><div class='price-change' style='background-color:rgba(255,255,255,0.05); color:{color}'>{p-prev:+.2f} ({(p-prev)/prev:+.2%})</div>{f'<div class="ext-price">🌙 {p_data["ext_label"]}: ${p_data["ext_price"]:.2f}</div>' if p_data['ext_price'] else ''}</div>", unsafe_allow_html=True)
    
    with st.spinner("🦁 正在调取机构底仓数据..."): heavy = fetch_financial_data_v105_2(ticker)
    h, i = heavy['history'], heavy['info']
    
    if not h.empty:
        # L-Box
        l_an = calculate_vision_analysis(h, i)
        if l_an:
            res_rows = "".join([f"<div class='l-item'><span>压力 ({p['d']})</span><span style='color:#fdba74'>{mk_range(p['v'])}<span class='tg-s'>{p['l']}</span></span></div>" for p in l_an['ress']])
            sup_rows = "".join([f"<div class='l-item'><span>支撑 ({p['d']})</span><span style='color:#86efac'>{mk_range(p['v'])}<span class='tg-s'>{p['l']}</span></span></div>" for p in l_an['sups']])
            st.markdown(f"<div class='l-box'><div class='l-title'>🦁 视野·交易计划 ({ticker})</div><div class='l-sub'>增速与估值</div><div class='l-item'><span>未来增速 (Rev)</span><span>{fmt_pct(l_an['growth'])}</span></div><div class='l-item'><span>合理估值 (25x-35x)</span><span style='font-weight:bold'>{l_an['val_range']}</span></div><div class='l-item'><span>技术面诊断</span><span style='font-weight:bold; color:#2563EB'>{l_an['tech']}</span></div><div class='l-sub'>关键点位</div>{res_rows}{sup_rows}</div>", unsafe_allow_html=True)

        with st.expander("🔗 商业版图 & 竞对 (思维导图)", expanded=False):
            ids, labels, parents = fetch_mind_map_data(ticker)
            fig_sun = go.Figure(go.Sunburst(ids=ids, labels=labels, parents=parents, branchvalues="total", marker=dict(colors=["#FF9F1C", "#ef4444", "#3b82f6", "#a855f7", "#10b981"])))
            fig_sun.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=400, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
            st.plotly_chart(fig_sun, use_container_width=True)

        with st.expander("🆚 跑赢大盘了吗? (收益对比)", expanded=False):
            if not heavy['compare'].empty:
                cmp = heavy['compare']
                fig2 = go.Figure(); fig2.add_trace(go.Scatter(x=cmp.index, y=cmp[ticker]*100, name=ticker, line=dict(width=3, color='#3b82f6'))); fig2.add_trace(go.Scatter(x=cmp.index, y=cmp['SP500']*100, name="SP500", line=dict(width=1.5, color='#9ca3af', dash='dot'))); fig2.add_trace(go.Scatter(x=cmp.index, y=cmp['Nasdaq']*100, name="Nasdaq", line=dict(width=1.5, color='#f97316', dash='dot')))
                fig2.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig2, use_container_width=True)

        bulls, bears = generate_bull_bear_thesis(h, i)
        with st.expander("🐂 vs 🐻 智能多空博弈 (Thesis)", expanded=False):
            c_bull, c_bear = st.columns(2)
            with c_bull: st.markdown(f"<div class='thesis-col thesis-bull'><b>🚀 多头逻辑</b><br>{'<br>'.join([f'✅ {b}' for b in bulls])}</div>", unsafe_allow_html=True)
            with c_bear: st.markdown(f"<div class='thesis-col thesis-bear'><b>🔻 空头逻辑</b><br>{'<br>'.join([f'⚠️ {b}' for b in bears])}</div>", unsafe_allow_html=True)

        with st.expander("📈 机构趋势图 (SuperTrend)", expanded=False):
            fig = go.Figure(); fig.add_trace(go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'], name='K线'))
            fig.add_trace(go.Scatter(x=h.index, y=h['ST_Lower'], mode='markers', marker=dict(color='orange', size=2), name='止损线'))
            fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📉 进阶指标 (Z-Score/ADX/CCI)", expanded=False):
            for name, col, clr, ref, info in [("乖离率 (Z-Score)", "Z_Score", "#f472b6", 2, "红线涨过头"), ("趋势强度 (ADX)", "ADX", "#fbbf24", 25, "25以上趋势强"), ("顺势指标 (CCI)", "CCI", "#22d3ee", 100, "100以上超买")]:
                st.markdown(f"##### {name}"); fig_z = go.Figure(); fig_z.add_trace(go.Scatter(x=h.index, y=h[col], line=dict(color=clr, width=1)))
                fig_z.add_hline(y=ref, line_dash='dot', line_color='red'); fig_z.add_hline(y=-ref, line_dash='dot', line_color='green')
                fig_z.update_layout(height=180, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig_z, use_container_width=True)
                st.markdown(f"<div class='ind-desc'>💡 <b>提示：</b> {info}</div>", unsafe_allow_html=True)

    # Core Data & Tabs
    st.subheader("📊 核心数据"); c1, c2, c3 = st.columns(3); safe_i = i if isinstance(i, dict) else {}
    c1.metric("市值", fmt_big(safe_i.get('marketCap'))); c2.metric("做空比", fmt_pct(safe_i.get('shortPercentOfFloat'))); c3.metric("股息率", fmt_pct(safe_i.get('dividendYield')))
    
    st.session_state.quant_score = calculate_quant_score(i, h)
    tabs = st.tabs(["📰 资讯", "👥 持仓 (深度)", "💰 估值", "🎓 深度研报"])
    
    with tabs[0]:
        news_df = process_news(heavy.get('news', []))
        if not news_df.empty: st.dataframe(news_df[['时间','标题','价格','链接']], column_config={"链接": st.column_config.LinkColumn("阅读"), "价格": st.column_config.TextColumn("🎯", width="small")}, hide_index=True)
        else: st.info("暂无新闻")
        
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🏦 机构持仓")
            if heavy.get('inst') is not None:
                # Top 5 only
                idf = heavy['inst'].copy().rename(columns={'Holder': '机构名称', 'pctHeld': '占比', 'Value': '市值'})
                st.dataframe(idf[['机构名称', '占比', '市值']], column_config={"占比": st.column_config.ProgressColumn("占比", format="%.2f%%", min_value=0, max_value=0.1)}, use_container_width=True, hide_index=True)
            else: st.info("暂无数据 (Yahoo 限制中)")
        with c2:
            st.subheader("🕴️ 内部交易")
            if heavy.get('insider') is not None:
                for index, row in heavy['insider'].head(10).iterrows():
                    # [NEW] Regex Parsing for Insider Text
                    trans_text = str(row.get('Text', ''))
                    # Default
                    action = "❓ 未知"; color = "#9ca3af"
                    if "Sale" in trans_text or "Sold" in trans_text: action = "🔴 减持"; color = "#ef4444"
                    elif "Purchase" in trans_text or "Buy" in trans_text: action = "🟢 增持"; color = "#4ade80"
                    
                    # Extract Price
                    price_match = re.search(r'price\s\$?(\d+\.?\d*)', trans_text)
                    price = f"${price_match.group(1)}" if price_match else "-"
                    
                    st.markdown(f"<div class='hold-card'><div><div class='hold-name'>{row.get('Insider')}</div><div class='hold-sub'>{row.get('Position')}</div></div><div style='text-align:right'><div style='color:{color};font-weight:bold'>{action} (均价 {price})</div><div class='hold-val'>{row.get('Shares')}股</div></div></div>", unsafe_allow_html=True)
            else: st.info("暂无数据 (Yahoo 限制中)")

    with tabs[2]:
        eps = safe_i.get('trailingEps', 0); bvps = safe_i.get('bookValue', 0); rt_p = p if p>0 else h['Close'].iloc[-1]
        if eps and bvps and rt_p: st.metric("格雷厄姆合理价", f"${(22.5 * eps * bvps) ** 0.5:.2f}", f"{( (22.5*eps*bvps)**0.5 - rt_p)/rt_p:.1%} Upside")
        st.markdown("---"); g = st.slider("预期增长率 %", 0, 50, 15)
        if eps: st.metric("DCF 估值", f"${(eps * ((1+g/100)**5) * 25) / (1.1**5):.2f}")

    with tabs[3]:
        st.header(f"🎓 {ticker} 深度研报"); st.markdown(f"<div class='report-text'>{safe_i.get('longBusinessSummary', '暂无描述')}</div>", unsafe_allow_html=True)
        gm, roe, peg = safe_i.get('grossMargins', 0), safe_i.get('returnOnEquity', 0), safe_i.get('pegRatio')
        st.markdown(f"<div class='guru-check'>{'✅' if gm>0.4 else '❌'} 毛利率 > 40% ({fmt_pct(gm)})</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='guru-check'>{'✅' if roe>0.15 else '❌'} ROE > 15% ({fmt_pct(roe)})</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='guru-check'>{'✅' if peg and peg < 1 else '❌'} 林奇 PEG < 1.0 (当前 {peg})</div>", unsafe_allow_html=True)

elif page == "🗓️ 财报地图":
    st.title("🗓️ 全行业财报热力图"); data = fetch_sector_earnings()
    if data:
        df = pd.DataFrame(data); fig = px.treemap(df, path=[px.Constant("全市场"), 'Sector', 'Code'], values=np.ones(len(df)), color='Days', color_continuous_scale='RdYlGn'); fig.update_traces(textinfo="label+text", texttemplate="%{label}<br>T-%{customdata[1]}"); fig.update_layout(height=600, template="plotly_dark", margin=dict(t=30, l=0, r=0, b=0)); st.plotly_chart(fig, use_container_width=True)
    else: st.info("数据更新中...")

else:
    st.title("📚 摩根·功能说明书 (Wiki)")
    st.markdown("""
    <div class='wiki-card'><div class='wiki-title'>1. 视野·交易计划 (Vision L-Box)</div><div class='wiki-text'>核心逻辑：L战法系统。通过均线、前高前低自动计算支撑位(S1)与压力位(R1)。</div></div>
    <div class='wiki-card'><div class='wiki-title'>2. 神奇九转 (TD Sequential)</div><div class='wiki-text'>原理：寻找衰竭点。红色9上涨力竭，绿色9下跌力竭。</div></div>
    <div class='wiki-card'><div class='wiki-title'>3. VWAP (机构线)</div><div class='wiki-text'>原理：机构持仓成本。股价在VWAP之上代表机构护盘。</div></div>
    """, unsafe_allow_html=True)