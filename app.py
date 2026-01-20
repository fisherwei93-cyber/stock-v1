import streamlit as st
import os

# ================= 0. 铁律配置 (V82.1: 找回消失的按钮) =================
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if key in os.environ:
        del os.environ[key]

st.set_page_config(page_title="摩根·V1 (Value)", layout="wide", page_icon="🦁")

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import re
import sys
import time

# 2. 样式死锁 (UI)
st.markdown("""
<style>
    /* 全局背景 */
    .stApp { background-color: #000000 !important; color: #FFFFFF !important; }
    section[data-testid="stSidebar"] { background-color: #111111 !important; }

    /* [FIX] 把顶部工具栏找回来，并隐藏不需要的菜单，只留箭头 */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        visibility: visible !important; /* 强制显示 */
    }
    /* 把侧边栏开关按钮染成醒目的橙色 */
    button[kind="header"] {
        color: #FF9F1C !important;
        font-weight: bold !important;
    }

    /* 指标高亮 */
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important; 
        font-size: 28px !important;
        font-weight: 900 !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #9CA3AF !important;
        font-size: 14px !important;
        font-weight: 700 !important;
    }
    
    /* 折叠栏优化 */
    .streamlit-expanderHeader {
        background-color: #222222 !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }
    .streamlit-expanderHeader p {
        color: #FFFFFF !important;
        font-size: 16px !important;
        font-weight: 700 !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: #FF9F1C !important;
        color: #FF9F1C !important;
    }

    /* 视野黄框 */
    .l-box {
        background-color: #FF9F1C;
        color: #000000 !important;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(255, 159, 28, 0.4);
        margin-bottom: 20px;
        border: 1px solid #e68a00;
        font-family: 'Segoe UI', sans-serif;
    }
    .l-title { font-size: 18px; font-weight: 900; border-bottom: 2px solid #000; padding-bottom: 8px; margin-bottom: 12px; color: #000; }
    .l-item { display: flex; justify-content: space-between; align-items: center; font-size: 14px; font-weight: 600; border-bottom: 1px dashed rgba(0,0,0,0.2); padding: 4px 0; color: #000; }
    
    /* 标签 */
    .tg-s { background: rgba(0,0,0,0.1); padding: 1px 5px; border-radius: 4px; font-size: 11px; margin-left: 6px; color: #333; }
    .tg-m { background: #fffbeb; padding: 1px 5px; border-radius: 4px; font-size: 11px; margin-left: 6px; color: #854d0e; border: 1px solid #eab308; }
    .tg-h { background: #000; color: #FF9F1C; padding: 1px 6px; border-radius: 4px; font-size: 11px; margin-left: 6px; font-weight: 800; }
    
    /* 评分卡 */
    .score-card { background: #1A1A1A; padding: 15px; border-radius: 12px; text-align: center; border: 1px solid #333; margin-bottom: 15px; }
    .sc-val { font-size: 42px; font-weight: 900; color: #4ade80; line-height: 1; }
    .sc-lbl { font-size: 12px; color: #D1D5DB; font-weight: bold; }
    
    /* 列表项 */
    .wl-row { background-color: #1A1A1A; padding: 12px; margin-bottom: 8px; border-radius: 6px; border-left: 4px solid #555; cursor: pointer; display: flex; justify-content: space-between; align-items: center; border: 1px solid #333; color: #FFFFFF; }
    .wl-row:hover { border-left-color: #FF9F1C; background-color: #2A2A2A; }
    
    .social-box { display: flex; gap: 10px; margin-top: 10px; }
    .sig-box { background: rgba(6, 78, 59, 0.8); border: 1px solid #065f46; padding: 10px; border-radius: 6px; margin-top: 10px; font-size: 13px; color: #fff; }
    .risk-box { background: rgba(127, 29, 29, 0.5); border: 1px solid #ef4444; padding: 10px; border-radius: 6px; margin-top: 10px; font-size: 13px; color: #fff; }
    .note-box { background: #1e1b4b; border-left: 4px solid #6366f1; padding: 10px; font-size: 12px; color: #e0e7ff; margin-top: 5px; border-radius: 4px; line-height: 1.6; }
    .teach-box { background: #422006; border-left: 4px solid #f97316; padding: 10px; font-size: 12px; color: #ffedd5; margin-top: 10px; border-radius: 4px; }
    .mc-box { background: #0f172a; border: 1px solid #1e293b; padding: 10px; border-radius: 6px; margin-top:5px; }
    
    .thesis-col { flex: 1; padding: 10px; border-radius: 6px; font-size: 13px; margin-top:5px; }
    .thesis-bull { background: rgba(6, 78, 59, 0.8); border: 1px solid #34d399; color: #fff; }
    .thesis-bear { background: rgba(127, 29, 29, 0.8); border: 1px solid #f87171; color: #fff; }
    
    /* 研报与Wiki样式 */
    .report-title { font-size: 22px; font-weight: 900; color: #FF9F1C; margin-bottom: 10px; border-left: 5px solid #FF9F1C; padding-left: 10px; }
    .report-text { font-size: 15px; line-height: 1.8; color: #E5E7EB; margin-bottom: 20px; background: #1A1A1A; padding: 15px; border-radius: 8px; }
    .guru-check { display: flex; align-items: center; margin-bottom: 8px; padding: 8px; background: #262626; border-radius: 6px; }
    
    .wiki-card { background: #1A1A1A; border: 1px solid #333; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
    .wiki-title { font-size: 20px; font-weight: bold; color: #FF9F1C; margin-bottom: 15px; border-bottom: 1px solid #444; padding-bottom: 5px; }
    .wiki-text { font-size: 14px; color: #E5E7EB; line-height: 1.8; margin-bottom: 10px; }
    .wiki-tag { background: #374151; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 5px; border: 1px solid #555; }
    
    /* 删除 header隐藏，改为显示 */
</style>
""", unsafe_allow_html=True)

# 字典 & 辅助函数
FAMOUS_INSTITUTIONS = {"Vanguard":"先锋", "Blackrock":"贝莱德", "Morgan Stanley":"大摩", "Goldman":"高盛", "Jpmorgan":"小摩", "Citadel":"城堡", "State Street":"道富", "Berkshire":"伯克希尔"}
RATING_MAP = {"Buy":"买入", "Hold":"持有", "Sell":"卖出", "Strong Buy":"强购", "Overweight":"增持", "Neutral":"中性", "Outperform":"跑赢"}
FIN_MAP = {
    "Total Revenue": "总营收", "Net Income": "净利润", "Gross Profit": "毛利润", "Operating Income": "营业利润", 
    "EBITDA": "EBITDA", "Total Expenses": "总支出", "Cost Of Revenue": "营收成本", "Basic EPS": "基本每股收益",
    "Diluted EPS": "稀释每股收益", "Total Assets": "总资产", "Total Liabilities Net Minority Interest": "总负债",
    "Total Equity Gross Minority Interest": "股东权益", "Free Cash Flow": "自由现金流", "Operating Cash Flow": "经营现金流"
}

def fmt_pct(v): return f"{v:.2%}" if isinstance(v, (int, float)) else "-"
def fmt_num(v): return f"{v:.2f}" if isinstance(v, (int, float)) else "-"
def fmt_big(v): 
    if not isinstance(v, (int, float)): return "-"
    if v > 1e12: return f"{v/1e12:.2f}T"
    if v > 1e9: return f"{v/1e9:.2f}B"
    if v > 1e6: return f"{v/1e6:.2f}M"
    return str(v)
def mk_range(v): 
    if not isinstance(v, (int, float)): return "-"
    return f"{v*0.985:.1f}-{v*1.015:.1f}"
def smart_translate(t, d): 
    if not isinstance(t, str): return t
    for k,v in d.items(): 
        if k.lower() in t.lower(): return v
    return t
def calculate_grade(val, type_):
    if val is None: return "N/A", "#94a3b8"
    if type_ == 'PE': return ("A+ 极低", "#10B981") if val < 20 else ("B 合理", "#3B82F6") if val < 40 else ("D 高估", "#EF4444")
    if type_ == 'Growth': return ("A+ 爆发", "#10B981") if val > 0.3 else ("B 稳健", "#3B82F6") if val > 0.1 else ("C 滞涨", "#EF4444")
    return "N/A", "#94a3b8"

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    HAS_YOUTUBE = True
except: HAS_YOUTUBE = False

if 'watchlist' not in st.session_state: st.session_state.watchlist = ['TSLA', 'NVDA', 'AAPL', 'AMD', 'PLTR']
if 'current_ticker' not in st.session_state: st.session_state.current_ticker = 'TSLA'

# ================= 2. 数据引擎 =================

@st.cache_data(ttl=300)
def fetch_stock_full_data(ticker):
    try:
        s = yf.Ticker(ticker)
        try: rt_price = s.fast_info.last_price
        except: rt_price = s.info.get('currentPrice', 0)
        
        h = s.history(period="2y") 
        if h.empty: raise Exception("Yahoo无数据")
        
        # --- [NEW] 黑科技指标计算 ---
        # 1. SuperTrend
        h['TR'] = np.maximum(h['High'] - h['Low'], np.abs(h['High'] - h['Close'].shift(1)))
        h['ATR'] = h['TR'].rolling(10).mean()
        multiplier = 3.0
        hl2 = (h['High'] + h['Low']) / 2
        h['ST_Upper'] = hl2 + (multiplier * h['ATR'])
        h['ST_Lower'] = hl2 - (multiplier * h['ATR'])
        
        # 2. Z-Score
        h['MA20'] = h['Close'].rolling(20).mean()
        h['STD20'] = h['Close'].rolling(20).std()
        h['Z_Score'] = (h['Close'] - h['MA20']) / h['STD20']
        
        # 3. Donchian
        h['DC_Upper'] = h['High'].rolling(20).max()
        h['DC_Lower'] = h['Low'].rolling(20).min()
        
        # 4. FVG
        h['FVG_Bull'] = (h['Low'] > h['High'].shift(2))
        h['FVG_Bear'] = (h['High'] < h['Low'].shift(2))

        # 5. VWAP
        v = h['Volume'].values
        tp = (h['High'] + h['Low'] + h['Close']) / 3
        h['VWAP'] = (tp * v).cumsum() / v.cumsum()

        # Williams %R
        lookback = 14
        hh = h['High'].rolling(lookback).max()
        ll = h['Low'].rolling(lookback).min()
        h['WR'] = -100 * (hh - h['Close']) / (hh - ll)

        # 基础指标
        exp12 = h['Close'].ewm(span=12, adjust=False).mean()
        exp26 = h['Close'].ewm(span=26, adjust=False).mean()
        h['MACD'] = exp12 - exp26
        h['Signal'] = h['MACD'].ewm(span=9, adjust=False).mean()
        h['Hist'] = h['MACD'] - h['Signal']
        
        delta = h['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        h['RSI'] = 100 - (100 / (1 + rs))
        
        low_min = h['Low'].rolling(9).min()
        high_max = h['High'].rolling(9).max()
        h['RSV'] = (h['Close'] - low_min) / (high_max - low_min) * 100
        h['K'] = h['RSV'].ewm(com=2).mean()
        h['D'] = h['K'].ewm(com=2).mean()
        h['J'] = 3 * h['K'] - 2 * h['D']
        
        h['OBV'] = (np.sign(h['Close'].diff()) * h['Volume']).fillna(0).cumsum()
        
        mfm = ((h['Close'] - h['Low']) - (h['High'] - h['Close'])) / (h['High'] - h['Low'])
        mfv = mfm * h['Volume']
        h['CMF'] = mfv.rolling(20).sum() / h['Volume'].rolling(20).sum()

        h['MA50'] = h['Close'].rolling(50).mean()
        h['MA60'] = h['Close'].rolling(60).mean()
        h['MA120'] = h['Close'].rolling(120).mean()
        h['MA200'] = h['Close'].rolling(200).mean()
        h['UPPER'] = h['MA20'] + 2*h['STD20']
        h['LOWER'] = h['MA20'] - 2*h['STD20']
        
        h['Peak'] = h['Close'].cummax()
        h['Drawdown'] = (h['Close'] - h['Peak']) / h['Peak']
        h['Vol_MA20'] = h['Volume'].rolling(20).mean()
        h['Whale'] = h['Volume'] > 2.0 * h['Vol_MA20']

        # TD 9
        h['TD'] = 0; h = h.copy(); c = h['Close'].values
        td_up = np.zeros(len(c)); td_down = np.zeros(len(c))
        for i in range(4, len(c)):
            if c[i] > c[i-4]: td_up[i] = td_up[i-1] + 1
            else: td_up[i] = 0
            if c[i] < c[i-4]: td_down[i] = td_down[i-1] + 1
            else: td_down[i] = 0
        h['TD_UP'] = td_up; h['TD_DOWN'] = td_down

        # Fibonacci
        max_p = h['High'].tail(120).max()
        min_p = h['Low'].tail(120).min()
        diff = max_p - min_p
        h['Fib_236'] = min_p + 0.236 * diff
        h['Fib_382'] = min_p + 0.382 * diff
        h['Fib_500'] = min_p + 0.5 * diff
        h['Fib_618'] = min_p + 0.618 * diff

        try:
            h_recent = h.iloc[-504:] 
            spy = yf.Ticker("SPY").history(period="2y")['Close']
            qqq = yf.Ticker("QQQ").history(period="2y")['Close']
            idx = h_recent.index.intersection(spy.index).intersection(qqq.index)
            cmp_df = pd.DataFrame({
                ticker: h_recent.loc[idx, 'Close'],
                "SP500": spy.loc[idx],
                "Nasdaq": qqq.loc[idx]
            })
            start = -252 if len(cmp_df)>252 else 0
            cmp_norm = cmp_df.iloc[start:] / cmp_df.iloc[start] - 1
        except: cmp_norm = pd.DataFrame()

        opt_data = None
        try:
            dates = s.options
            if dates:
                near_date = dates[0]
                opt = s.option_chain(near_date)
                opt_data = {"date": near_date, "calls": opt.calls, "puts": opt.puts}
        except: pass

        # [FIX] 钛合金防爆门
        safe_info = s.info if s.info is not None else {}
        
        return {
            "history": h, "info": safe_info, "rt_price": rt_price,
            "news": s.news, "upgrades": s.upgrades_downgrades,
            "fin": s.quarterly_financials, "inst": s.institutional_holders, "insider": s.insider_transactions,
            "compare": cmp_norm, "options": opt_data,
            "error": None
        }
    except Exception as e:
        dates = pd.date_range(end=datetime.datetime.today(), periods=50)
        df = pd.DataFrame({'Open':100,'Close':100,'High':100,'Low':100,'Volume':0}, index=dates)
        return {
            "history":df, "info":{}, "rt_price":0, "news":[], "error": str(e), 
            "compare":pd.DataFrame(), "options":None, 
            "upgrades":None, "fin":None, "inst":None, "insider":None
        }

@st.cache_data(ttl=3600)
def fetch_macro_data():
    try:
        tickers = ["^VIX", "^TNX", "DX-Y.NYB"] 
        data = yf.download(tickers, period="5d", progress=False)['Close'].iloc[-1]
        return data
    except: return None

@st.cache_data(ttl=3600)
def fetch_correlation_data(ticker):
    try:
        benchmarks = ['SPY', 'QQQ', 'GLD', 'BTC-USD']
        data = yf.download([ticker] + benchmarks, period="1y", progress=False)['Close']
        if data.empty: return None
        data = data.pct_change().dropna()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        if ticker not in data.columns: return None
        return data.corrwith(data[ticker]).drop(ticker)
    except: return None

@st.cache_data(ttl=60)
def fetch_related_tickers(ticker, info):
    relations = {
        "NVDA": ["AMD", "TSM", "SMH", "ARM", "INTC"], 
        "TSLA": ["NIO", "XPEV", "LCID", "RIVN", "BYDDF"],
        "AAPL": ["MSFT", "GOOG", "AMZN", "META", "QCOM"], 
        "AMD": ["NVDA", "INTC", "TSM", "AVGO"],
        "BABA": ["PDD", "JD", "BIDU", "KWEB", "TCEHY"], 
        "PLTR": ["AI", "SNOW", "DDOG", "CRWD"],
        "META": ["GOOG", "SNAP", "PINS", "TTD"], 
        "AMZN": ["BABA", "WMT", "EBAY", "SHOP"]
    }
    return relations.get(ticker, [])

@st.cache_data(ttl=60)
def fetch_watchlist_snapshot(tickers):
    data = []
    for t in tickers:
        try:
            s = yf.Ticker(t)
            p = s.fast_info.last_price; prev = s.fast_info.previous_close
            chg = (p - prev) / prev
            data.append({"sym":t, "p":p, "chg":chg})
        except: data.append({"sym":t, "p":0, "chg":0})
    return data

# [ALGO] 视野逻辑 5.1
def calculate_vision_analysis(df, info):
    if len(df) < 250: return None
    curr = df['Close'].iloc[-1]
    
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    ma60 = df['Close'].rolling(60).mean().iloc[-1]
    ma120 = df['Close'].rolling(120).mean().iloc[-1]
    ma200 = df['Close'].rolling(200).mean().iloc[-1]
    low_60 = df['Low'].tail(60).min(); high_60 = df['High'].tail(60).max()
    low_52w = df['Low'].tail(250).min(); high_52w = df['High'].tail(250).max()
    high_20 = df['High'].tail(20).max()
    
    pts = []
    if curr > ma20: pts.append({"t":"sup", "l":"小", "v":ma20, "d":"MA20/月线"})
    if curr > ma60: pts.append({"t":"sup", "l":"中", "v":ma60, "d":"MA60/趋势"})
    if curr > low_60: pts.append({"t":"sup", "l":"强", "v":low_60, "d":"箱体底/前低"})
    if curr > ma120: pts.append({"t":"sup", "l":"强", "v":ma120, "d":"MA120/半年线"})
    if curr > ma200: pts.append({"t":"sup", "l":"超强", "v":ma200, "d":"MA200/年线"})
    if curr > low_52w: pts.append({"t":"sup", "l":"超强", "v":low_52w, "d":"52周低"})
    if curr < ma20: pts.append({"t":"res", "l":"小", "v":ma20, "d":"MA20/反压"})
    if curr < high_20: pts.append({"t":"res", "l":"小", "v":high_20, "d":"短期前高"})
    if curr < ma60: pts.append({"t":"res", "l":"中", "v":ma60, "d":"MA60"})
    if curr < high_60: pts.append({"t":"res", "l":"强", "v":high_60, "d":"箱体顶/套牢区"})
    if curr < high_52w: pts.append({"t":"res", "l":"超强", "v":high_52w, "d":"52周高/历史顶"})
    
    def filter_pts(p_list, reverse=False):
        p_list = sorted(p_list, key=lambda x:x['v'], reverse=reverse)
        res = []
        if p_list:
            res.append(p_list[0])
            for p in p_list[1:]:
                if abs(p['v'] - res[-1]['v']) / res[-1]['v'] > 0.02: res.append(p)
                else:
                    lv_map = {"小":1,"中":2,"强":3,"超强":4}
                    if lv_map[p['l']] > lv_map[res[-1]['l']]: res[-1] = p
        return res[:3]

    sups = filter_pts([p for p in pts if p['t']=="sup"], reverse=True)
    ress = filter_pts([p for p in pts if p['t']=="res"], reverse=False)
    
    # [FIX] 安全字典
    if not isinstance(info, dict): info = {}
    eps_fwd = info.get('forwardEps'); val_data = f"{eps_fwd*25:.0f}-{eps_fwd*35:.0f} (25x-35x)" if eps_fwd else "N/A"
    
    rsi = df['RSI'].iloc[-1]; macd_val = df['MACD'].iloc[-1]
    tech = []
    if rsi > 70: tech.append(f"RSI超买({rsi:.0f})")
    elif rsi < 30: tech.append(f"RSI超卖({rsi:.0f})")
    else: tech.append(f"RSI中性({rsi:.0f})")
    if macd_val > 0: tech.append("MACD多头")
    else: tech.append("MACD空头")
    
    return {"growth": info.get('revenueGrowth', 0), "val_range": val_data, "sups": sups, "ress": ress, "tech": " | ".join(tech)}

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

def process_news(news_list):
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

def enrich_upgrades(upgrades, news_df):
    if upgrades is None or news_df.empty: return upgrades
    upgrades = upgrades.copy()
    upgrades['新闻目标价'] = "-"
    valid = news_df[news_df['价格'] != "-"].head(30)
    for idx, row in upgrades.head(20).iterrows():
        firm = str(row['Firm']).split(' ')[0].lower()
        if len(firm) < 3: continue
        for _, n in valid.iterrows():
            if firm in str(n['标题']).lower():
                upgrades.at[idx, '新闻目标价'] = f"✅ {n['价格']}"
                break
    return upgrades

def calculate_max_pain(calls, puts):
    if calls.empty or puts.empty: return 0
    strikes = sorted(set(calls['strike']).union(set(puts['strike'])))
    min_loss = float('inf'); max_pain = 0
    for s in strikes:
        loss = 0
        c_loss = calls[calls['strike'] < s].apply(lambda x: (s - x['strike']) * x['openInterest'], axis=1).sum()
        p_loss = puts[puts['strike'] > s].apply(lambda x: (x['strike'] - s) * x['openInterest'], axis=1).sum()
        loss = c_loss + p_loss
        if loss < min_loss:
            min_loss = loss; max_pain = s
    return max_pain

def calculate_seasonality(df):
    if df.empty: return None
    df = df.copy()
    df['Month'] = df.index.month
    df['Ret'] = df['Close'].pct_change()
    monthly_stats = df.groupby('Month')['Ret'].agg(['mean', lambda x: (x>0).sum() / len(x)])
    monthly_stats.columns = ['Avg Return', 'Win Rate']
    return monthly_stats

def calculate_volume_profile(df, bins=50):
    price_min = df['Low'].min()
    price_max = df['High'].max()
    hist = np.histogram(df['Close'], bins=bins, range=(price_min, price_max), weights=df['Volume'])
    return hist[1][:-1], hist[0]

def generate_bull_bear_thesis(df, info):
    if df.empty: return [], []
    bulls = []; bears = []
    if 'Close' not in df.columns: return [], []
    curr = df['Close'].iloc[-1]; ma200 = df['MA200'].iloc[-1]; rsi = df['RSI'].iloc[-1]
    if curr > ma200: bulls.append("股价站上年线 (长期牛市)")
    else: bears.append("股价跌破年线 (长期熊市)")
    if rsi < 30: bulls.append("RSI超卖 (反弹预期)")
    if rsi > 70: bears.append("RSI超买 (回调风险)")
    if not isinstance(info, dict): info = {}
    short = info.get('shortPercentOfFloat', 0)
    if short and short > 0.2: bulls.append("逼空潜力大 (Short Squeeze)")
    if short and short > 0.15: bears.append("做空拥挤 (机构看空)")
    while len(bulls) < 3: bulls.append("暂无明显多头信号")
    while len(bears) < 3: bears.append("暂无明显空头信号")
    return bulls[:3], bears[:3]

# [NEW] Documentation
def render_documentation():
    st.title("📚 摩根·功能说明书 (Wiki)")
    
    st.markdown("""
    <div class='wiki-card'>
        <div class='wiki-title'>1. 视野·交易计划 (Vision L-Box)</div>
        <div class='wiki-text'>
            <b>核心逻辑：</b> 基于“L战法”的支撑压力系统。<br>
            <span class='wiki-tag'>R1/R2 (Resistance)</span>：压力位。股价涨到这里容易被打下来，是卖出或减仓的参考点。<br>
            <span class='wiki-tag'>S1/S2 (Support)</span>：支撑位。股价跌到这里容易反弹，是买入或补仓的参考点。<br>
            <b>黄框</b>：是整个系统的大脑，直接告诉你现在是该买还是该卖。
        </div>
    </div>
    
    <div class='wiki-card'>
        <div class='wiki-title'>2. 神奇九转 (TD Sequential)</div>
        <div class='wiki-text'>
            <b>原理：</b> 这是一个专门用来抓“顶”和“底”的计数器。<br>
            <span style='color:#f87171'><b>红色 9 (高9)</b></span>：当连续9根K线的收盘价都高于4天前的收盘价时触发。意味着上涨动能耗尽，大概率要回调。<b>（卖点）</b><br>
            <span style='color:#4ade80'><b>绿色 9 (低9)</b></span>：当连续9根K线的收盘价都低于4天前的收盘价时触发。意味着下跌动能衰竭，大概率要反弹。<b>（买点）</b>
        </div>
    </div>
    
    <div class='wiki-card'>
        <div class='wiki-title'>3. VWAP (成交量加权平均价)</div>
        <div class='wiki-text'>
            <b>原理：</b> 这是机构交易员的“生命线”。它不仅仅是平均价，还考虑了成交量。<br>
            <b>用法：</b><br>
            - 股价 > VWAP：说明今天买入的人大部分是赚钱的，市场强势，机构在护盘。<br>
            - 股价 < VWAP：说明今天买入的人大部分被套了，市场弱势，机构在出货。
        </div>
    </div>
    
    <div class='wiki-card'>
        <div class='wiki-title'>4. 蒙特卡洛预测 (Monte Carlo)</div>
        <div class='wiki-text'>
            <b>原理：</b> 计算机通过模拟未来 30 天的 100 种可能走势。<br>
            <b>计算根据：</b> 使用股票过去 1 年的“历史波动率” (Volatility) 和“漂移率” (Drift, 平均涨幅)，通过“几何布朗运动”公式进行随机游走模拟。<br>
            <b>判断方法：</b><br>
            - <b>悲观底线 (Bottom 5%)</b>：有 95% 的概率股价会高于此线。适合做止损位。<br>
            - <b>乐观高点 (Top 5%)</b>：只有 5% 的概率股价能涨破此线。适合做止盈位。<br>
            - <b>中位数 (Median)</b>：最可能的价格路径。
        </div>
    </div>
    
    <div class='wiki-card'>
        <div class='wiki-title'>5. 基本面六维雷达 (Fundamental Spider)</div>
        <div class='wiki-text'>
            <b>原理：</b> 快速体检表。<br>
            - <b>PE (反向)</b>：越靠外圈，估值越便宜。<br>
            - <b>Rev Growth</b>：营收增长越快越好。<br>
            - <b>Short Ratio</b>：做空越少越安全。<br>
            <b>形状：</b> 图形越饱满（面积越大），公司基本面越完美，越像“六边形战士”。
        </div>
    </div>
    
    <div class='wiki-card'>
        <div class='wiki-title'>6. 🦸‍♂️ SuperTrend (超级趋势)</div>
        <div class='wiki-text'>
            <b>原理：</b> 基于 ATR 波动率的趋势跟踪系统。<br>
            <b>用法：</b> 图表上那条变色的线。<b>绿色</b>代表处于上涨趋势（持股），<b>红色</b>代表处于下跌趋势（空仓）。它是最好的<b>“移动止损线”</b>。
        </div>
    </div>
    
    <div class='wiki-card'>
        <div class='wiki-title'>7. 🕳️ FVG (聪明钱缺口)</div>
        <div class='wiki-text'>
            <b>原理：</b> 机构暴力拉升或砸盘时留下的“真空地带”。<br>
            <b>用法：</b> 图中的<b>紫色方块</b>。股价通常会像有磁铁一样，回踩这些方块去“填补缺口”。如果你错过了第一波行情，可以在 FVG 区域挂单等回调。
        </div>
    </div>
    
    <div class='wiki-card'>
        <div class='wiki-title'>8. 📊 Z-Score (统计学乖离率)</div>
        <div class='wiki-text'>
            <b>原理：</b> 统计学上的“标准差”。<br>
            <b>用法：</b> 告诉你股价偏离均线有多远。如果 Z-Score 超过 +2，说明“涨过头了”，随时可能回调；如果低于 -2，说明“跌过头了”，随时可能反弹。
        </div>
    </div>
    
    <div class='wiki-card'>
        <div class='wiki-title'>9. 🐢 唐奇安通道 (Donchian Channels)</div>
        <div class='wiki-text'>
            <b>原理：</b> 海龟交易法则的核心。<br>
            <b>用法：</b> 突破上轨（过去20天最高价）是<b>买入信号</b>；跌破下轨（过去20天最低价）是<b>卖出信号</b>。做大趋势的神器。
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================= 6. 页面路由 =================
page = st.sidebar.radio("📌 导航", ["🚀 股票分析", "📖 功能说明书"])

if page == "🚀 股票分析":
    with st.sidebar:
        with st.expander("📺 视频分析 (YouTube)", expanded=True):
            yt_url = st.text_input("视频链接", placeholder="粘贴URL...")
            if st.button("🚀 提取 Prompt"):
                try:
                    from youtube_transcript_api import YouTubeTranscriptApi
                    vid = yt_url.split("v=")[-1].split("&")[0]
                    t = YouTubeTranscriptApi.get_transcript(vid, languages=['zh-Hans','en'])
                    txt = " ".join([x['text'] for x in t])
                    st.text_area("复制:", f"我是基金经理。分析此视频：\n1.核心观点\n2.提及股票\n3.多空判断\n\n内容：{txt[:6000]}...", height=150)
                except Exception as e: st.error(f"提取失败: {e}")

        st.markdown("---")
        if 'quant_score' in st.session_state:
            s, n = st.session_state.quant_score
            c = "#4ade80" if s>=60 else "#f87171"
            st.markdown(f"<div class='score-card'><div class='sc-lbl'>MORGAN SCORE</div><div class='sc-val' style='color:{c}'>{s}</div><div class='sc-lbl' style='color:#9CA3AF'>{n}</div></div>", unsafe_allow_html=True)
        
        # 实时数据
        ticker = st.session_state.current_ticker
        with st.spinner(f"🦁 正在连接华尔街数据源: {ticker} ..."):
            data = fetch_stock_full_data(ticker)
        
        if data['error']:
            st.error("数据获取失败，请重试")
            i = {}
        else:
            i = data['info']
            h = data['history']

        if i:
            st.caption("📊 实时数据")
            c1, c2 = st.columns(2)
            c1.metric("市值", fmt_big(i.get('marketCap')))
            c2.metric("Beta", fmt_num(i.get('beta')))
            
            if not h.empty:
                atr = h['ATR'].iloc[-1]
                curr_p = i.get('currentPrice', h['Close'].iloc[-1])
                stop_loss = curr_p - (2 * atr)
                dd_curr = h['Drawdown'].iloc[-1]
                dd_max = h['Drawdown'].min()
                st.markdown(f"""
                <div class='risk-box'>
                    <b>🛡️ 风控助手 (ATR动态止损)</b><br>
                    当前波动率: {atr:.2f}<br>
                    建议止损位: <span style='color:#f87171;font-weight:bold'>${stop_loss:.2f}</span><br>
                    <hr style='margin:5px 0; border-color:#7f1d1d'>
                    当前回撤: {dd_curr:.1%}<br>
                    52周最大回撤: <b>{dd_max:.1%}</b>
                </div>
                """, unsafe_allow_html=True)
                
                # [NEW] 凯利公式计算器
                with st.expander("🧮 凯利仓位计算器"):
                    win_prob = st.slider("胜率 (%)", 0, 100, 50)
                    risk_reward = st.slider("盈亏比 (1:x)", 1.0, 5.0, 2.0)
                    # Kelly = P - (1-P)/R
                    P = win_prob / 100
                    R = risk_reward
                    kelly = P - (1-P)/R
                    if kelly > 0:
                        st.markdown(f"建议仓位: <b style='color:#4ade80'>{kelly:.1%}</b>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"建议仓位: <b style='color:#f87171'>不参与 (0%)</b>", unsafe_allow_html=True)

            rel_tickers = fetch_related_tickers(ticker, i)
            if rel_tickers:
                st.markdown("---")
                st.caption("🔗 产业链联动")
                rel_data = fetch_watchlist_snapshot(rel_tickers)
                for r in rel_data:
                    rc = "#4ade80" if r['chg']>=0 else "#f87171"
                    c_btn, c_txt = st.columns([1, 1.5])
                    with c_btn:
                        if st.button(r['sym'], key=f"btn_{r['sym']}"):
                            st.session_state.current_ticker = r['sym']
                            st.rerun()
                    with c_txt:
                        st.markdown(f"<div style='margin-top:5px; font-size:13px; color:{rc}'>{r['chg']:.2%}</div>", unsafe_allow_html=True)

        st.caption("我的自选")
        c1, c2 = st.columns([3,1])
        new_t = c1.text_input("Code", label_visibility="collapsed").upper()
        if c2.button("➕") and new_t:
            if new_t not in st.session_state.watchlist: st.session_state.watchlist.append(new_t); st.rerun()
        
        wl = fetch_watchlist_snapshot(st.session_state.watchlist)
        for item in wl:
            sym = item['sym']; p = item['p']; chg = item['chg']
            c_val = "#4ade80" if chg >= 0 else "#f87171"
            st.markdown(f"<div class='wl-row' style='border-left-color: {c_val}'><div style='font-weight:bold;'>{sym}</div><div style='text-align:right'><div style='font-family:monospace; font-weight:bold;'>{p:.2f}</div><div style='font-size:11px; color:{c_val};'>{chg:.2%}</div></div></div>", unsafe_allow_html=True)
            cols = st.columns(2)
            if cols[0].button("分析", key=f"a_{sym}"): st.session_state.current_ticker = sym; st.rerun()
            if cols[1].button("删", key=f"d_{sym}"): st.session_state.watchlist.remove(sym); st.rerun()

    render_main_app()

else:
    render_documentation()