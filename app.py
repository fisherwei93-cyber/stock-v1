import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import datetime
import re
import sys
import time

# ================= 0. 铁律配置 (版本1 基石) =================
# 这里的代理设置在本地运行有效，部署到云端会自动忽略
PROXY_URL = "http://127.0.0.1:8080"
if os.environ.get("STREAMLIT_SERVER_PORT"): # 简单判断是否在云端
    pass
else:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL

st.set_page_config(page_title="摩根·V1 (Pro)", layout="wide", page_icon="🦁")

# 2. 样式死锁
st.markdown("""
<style>
    .stApp { background-color: #F0F2F6; }
    
    /* 视野黄框 */
    .l-box {
        background-color: #FF9F1C;
        color: #000000;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        font-family: 'Segoe UI', sans-serif;
        border: 1px solid #e68a00;
    }
    .l-title { font-size: 18px; font-weight: 900; border-bottom: 2px solid #000; padding-bottom: 8px; margin-bottom: 12px; }
    .l-sub { font-size: 15px; font-weight: 800; text-decoration: underline; margin-top: 15px; margin-bottom: 8px; }
    .l-item { display: flex; justify-content: space-between; align-items: center; font-size: 14px; font-weight: 600; border-bottom: 1px dashed rgba(0,0,0,0.2); padding: 4px 0; }
    
    /* 标签系统 */
    .tg-s { background: rgba(255,255,255,0.5); padding: 1px 5px; border-radius: 4px; font-size: 11px; margin-left: 6px; color: #333; }
    .tg-m { background: #fef08a; padding: 1px 5px; border-radius: 4px; font-size: 11px; margin-left: 6px; color: #854d0e; border: 1px solid #eab308; }
    .tg-h { background: #000; color: #FF9F1C; padding: 1px 6px; border-radius: 4px; font-size: 11px; margin-left: 6px; font-weight: 800; }
    
    /* 评分卡 */
    .score-card { background: white; padding: 15px; border-radius: 10px; text-align: center; border: 2px solid #e2e8f0; margin-bottom: 15px; }
    .sc-val { font-size: 42px; font-weight: 900; color: #2563EB; line-height: 1; }
    .sc-lbl { font-size: 12px; color: #64748b; font-weight: bold; }
    
    /* 自选股 & 关联股 */
    .wl-row { background: white; padding: 8px 10px; margin-bottom: 5px; border-radius: 4px; border-left: 4px solid #ddd; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }
    .wl-row:hover { border-left-color: #2563EB; background: #eef2ff; }
    .rel-btn { width: 100%; text-align: left; padding: 5px; margin: 2px 0; border: 1px solid #eee; border-radius: 4px; background: white; cursor: pointer; }
    .rel-btn:hover { background: #f0f9ff; border-color: #3b82f6; }
    
    /* 社交按钮 */
    .social-box { display: flex; gap: 10px; margin-top: 10px; }
    
    /* 信号/风控/教学 */
    .sig-box { background: #f0fdf4; border: 1px solid #bbf7d0; padding: 10px; border-radius: 6px; margin-top: 10px; font-size: 13px; color: #166534; }
    .risk-box { background: #fff1f2; border: 1px solid #fecaca; padding: 10px; border-radius: 6px; margin-top: 10px; font-size: 13px; color: #9f1239; }
    .note-box { background: #eef2ff; border-left: 4px solid #6366f1; padding: 10px; font-size: 12px; color: #4338ca; margin-top: 5px; border-radius: 4px; line-height: 1.6; }
    .teach-box { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 10px; font-size: 12px; color: #92400e; margin-top: 10px; border-radius: 4px; }
    
    .thesis-col { flex: 1; padding: 10px; border-radius: 6px; font-size: 13px; margin-top:5px; }
    .thesis-bull { background: #f0fdf4; border: 1px solid #86efac; color: #166534; }
    .thesis-bear { background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; }
    
    header {visibility: hidden;}
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

# ================= 2. 数据引擎 (优化缓存) =================

@st.cache_data(ttl=300)
def fetch_stock_full_data(ticker):
    try:
        s = yf.Ticker(ticker)
        try: rt_price = s.fast_info.last_price
        except: rt_price = s.info.get('currentPrice', 0)
        
        h = s.history(period="5y") 
        if h.empty: raise Exception("Yahoo无数据")
        
        # --- 指标计算 ---
        # MACD
        exp12 = h['Close'].ewm(span=12, adjust=False).mean()
        exp26 = h['Close'].ewm(span=26, adjust=False).mean()
        h['MACD'] = exp12 - exp26
        h['Signal'] = h['MACD'].ewm(span=9, adjust=False).mean()
        h['Hist'] = h['MACD'] - h['Signal']
        # RSI
        delta = h['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        h['RSI'] = 100 - (100 / (1 + rs))
        # KDJ
        low_min = h['Low'].rolling(9).min()
        high_max = h['High'].rolling(9).max()
        h['RSV'] = (h['Close'] - low_min) / (high_max - low_min) * 100
        h['K'] = h['RSV'].ewm(com=2).mean()
        h['D'] = h['K'].ewm(com=2).mean()
        h['J'] = 3 * h['K'] - 2 * h['D']
        # OBV
        h['OBV'] = (np.sign(h['Close'].diff()) * h['Volume']).fillna(0).cumsum()
        # MA & BOLL & ATR
        h['TR'] = np.maximum(h['High'] - h['Low'], np.abs(h['High'] - h['Close'].shift(1)))
        h['ATR'] = h['TR'].rolling(14).mean()
        h['MA20'] = h['Close'].rolling(20).mean()
        h['MA50'] = h['Close'].rolling(50).mean()
        h['MA60'] = h['Close'].rolling(60).mean()
        h['MA120'] = h['Close'].rolling(120).mean()
        h['MA200'] = h['Close'].rolling(200).mean()
        h['STD20'] = h['Close'].rolling(20).std()
        h['UPPER'] = h['MA20'] + 2*h['STD20']
        h['LOWER'] = h['MA20'] - 2*h['STD20']
        # Drawdown & Whale
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

        # 对比数据
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

        return {
            "history": h, "info": s.info, "rt_price": rt_price,
            "news": s.news, "upgrades": s.upgrades_downgrades,
            "fin": s.quarterly_financials, "inst": s.institutional_holders, "insider": s.insider_transactions,
            "compare": cmp_norm, "options": opt_data,
            "error": None
        }
    except Exception as e:
        dates = pd.date_range(end=datetime.datetime.today(), periods=50)
        df = pd.DataFrame({'Open':100,'Close':100,'High':100,'Low':100,'Volume':0}, index=dates)
        return {"history":df, "info":{}, "rt_price":0, "news":[], "error": str(e), "compare":pd.DataFrame(), "options":None}

@st.cache_data(ttl=3600)
def fetch_macro_data():
    try:
        tickers = ["^VIX", "^TNX", "DX-Y.NYB"] 
        data = yf.download(tickers, period="5d", progress=False)['Close'].iloc[-1]
        return data
    except: return None

@st.cache_data(ttl=3600)
def fetch_related_tickers(ticker, info):
    # 手动维护的热门关联库，比自动获取更准
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
    # 如果不在库里，尝试返回同行业的头
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
    
    high_20 = df['High'].tail(20).max(); low_20 = df['Low'].tail(20).min()
    high_60 = df['High'].tail(60).max(); low_60 = df['Low'].tail(60).min()
    high_52w = df['High'].tail(250).max(); low_52w = df['Low'].tail(250).min()
    
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
    bulls = []; bears = []
    curr = df['Close'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]; ma200 = df['MA200'].iloc[-1]; rsi = df['RSI'].iloc[-1]
    if curr > ma200: bulls.append("股价站上年线 (长期牛市)")
    else: bears.append("股价跌破年线 (长期熊市)")
    if ma50 > ma200: bulls.append("均线金叉 (多头排列)")
    if rsi < 30: bulls.append("RSI超卖 (反弹预期)")
    if rsi > 70: bears.append("RSI超买 (回调风险)")
    peg = info.get('pegRatio'); short = info.get('shortPercentOfFloat', 0)
    if peg and peg < 1.0: bulls.append(f"PEG低估 ({peg})")
    if peg and peg > 2.5: bears.append(f"PEG高估 ({peg})")
    if short > 0.2: bulls.append("逼空潜力大 (Short Squeeze)")
    if short > 0.15: bears.append("做空拥挤 (机构看空)")
    while len(bulls) < 3: bulls.append("暂无明显多头信号")
    while len(bears) < 3: bears.append("暂无明显空头信号")
    return bulls[:3], bears[:3]

# ================= 3. 主程序逻辑 (V1.1) =================

ticker = st.session_state.current_ticker

# [SPEED] 提前获取，且加 Loading 动画
with st.spinner(f"🦁 正在连接华尔街数据源: {ticker} ..."):
    data = fetch_stock_full_data(ticker)

if data['error']:
    st.error(f"数据获取失败: {data['error']}")
    h, i = pd.DataFrame(), {}
else:
    h, i = data['history'], data['info']

# 预计算
if not h.empty:
    rt_price = data['rt_price']
    prev = h['Close'].iloc[-1]
    chg = (rt_price - prev)/prev
    st.session_state.quant_score = calculate_quant_score(i, h)
    l_an = calculate_vision_analysis(h, i)
else:
    rt_price, chg, l_an = 0, 0, None

# 渲染侧边栏 (使用最新鲜的数据)
with st.sidebar:
    st.title("🦁 摩根·V1 (Pro)")
    
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
        c = "#16a34a" if s>=60 else "#dc2626"
        st.markdown(f"<div class='score-card'><div class='sc-lbl'>MORGAN SCORE</div><div class='sc-val' style='color:{c}'>{s}</div><div class='sc-lbl' style='color:#64748b'>{n}</div></div>", unsafe_allow_html=True)
    
    if i:
        st.caption("📊 实时数据")
        c1, c2 = st.columns(2)
        c1.metric("市值", fmt_big(i.get('marketCap')))
        c2.metric("Beta", fmt_num(i.get('beta')))
        
        if not h.empty:
            signals = []
            curr = h['Close'].iloc[-1]
            ma20 = h['MA20'].iloc[-1]; ma60 = h['MA60'].iloc[-1]
            vol_curr = h['Volume'].iloc[-1]; vol_avg = h['Volume'].tail(10).mean()
            
            if curr > ma20 and ma20 > ma60: signals.append("🐂 多头排列")
            if vol_curr > 1.8 * vol_avg: signals.append("🔥 放量异动")
            if h['RSI'].iloc[-1] > 70: signals.append("⚠️ RSI超买")
            if curr > h['UPPER'].iloc[-1]: signals.append("🚀 突破布林")
            if h['TD_UP'].iloc[-1] == 9: signals.append("🛑 神奇九转(高9)")
            if h['TD_DOWN'].iloc[-1] == 9: signals.append("💎 神奇九转(低9)")
            
            if signals: st.markdown(f"<div class='sig-box'>{' | '.join(signals)}</div>", unsafe_allow_html=True)
            else: st.markdown(f"<div class='sig-box' style='color:gray'>暂无明显技术信号</div>", unsafe_allow_html=True)

            atr = h['ATR'].iloc[-1]
            curr_p = i.get('currentPrice', h['Close'].iloc[-1])
            stop_loss = curr_p - (2 * atr)
            dd_curr = h['Drawdown'].iloc[-1]
            dd_max = h['Drawdown'].min()
            st.markdown(f"""
            <div class='risk-box'>
                <b>🛡️ 风控助手 (ATR动态止损)</b><br>
                当前波动率: {atr:.2f}<br>
                建议止损位: <span style='color:#dc2626;font-weight:bold'>${stop_loss:.2f}</span><br>
                <hr style='margin:5px 0'>
                当前回撤: {dd_curr:.1%}<br>
                52周最大回撤: <b>{dd_max:.1%}</b>
            </div>
            """, unsafe_allow_html=True)
            
        tgt_low = i.get('targetLowPrice'); tgt_high = i.get('targetHighPrice'); curr_p = i.get('currentPrice', 0)
        if tgt_low and tgt_high and curr_p:
            st.caption("🎯 华尔街目标价区间")
            progress = (curr_p - tgt_low) / (tgt_high - tgt_low)
            st.progress(max(0, min(1, progress)))
            c3, c4 = st.columns(2)
            c3.caption(f"低: ${tgt_low}"); c4.caption(f"高: ${tgt_high}")
            
        # [INTERACTION] 关联资产点击跳转
        rel_tickers = fetch_related_tickers(ticker, i)
        if rel_tickers:
            st.markdown("---")
            st.caption("🔗 产业链联动 (点击切换)")
            rel_data = fetch_watchlist_snapshot(rel_tickers)
            for r in rel_data:
                rc = "green" if r['chg']>=0 else "red"
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
        c_val = "#16a34a" if chg >= 0 else "#dc2626"
        st.markdown(f"<div class='wl-row' style='border-left-color: {c_val}'><div style='font-weight:bold;'>{sym}</div><div style='text-align:right'><div style='font-family:monospace; font-weight:bold;'>{p:.2f}</div><div style='font-size:11px; color:{c_val};'>{chg:.2%}</div></div></div>", unsafe_allow_html=True)
        cols = st.columns(2)
        if cols[0].button("分析", key=f"a_{sym}"): st.session_state.current_ticker = sym; st.rerun()
        if cols[1].button("删", key=f"d_{sym}"): st.session_state.watchlist.remove(sym); st.rerun()

c_main, c_fac = st.columns([2, 3])
with c_main:
    st.metric(f"{ticker} 实时", f"${rt_price:.2f}", f"{chg:.2%}")
    st.caption(f"{i.get('longName')} | {i.get('industry')}")
    st.markdown("<div class='social-box'>", unsafe_allow_html=True)
    c_btn = st.columns(4)
    c_btn[0].link_button("🔥 谷歌搜", f"https://www.google.com/search?q=why+is+{ticker}+stock+moving+today")
    c_btn[1].link_button("🎯 目标价", f"https://www.google.com/search?q={ticker}+stock+target+price")
    c_btn[2].link_button("👽 Reddit", f"https://www.reddit.com/search/?q=${ticker}")
    c_btn[3].link_button("🐦 Twitter", f"https://twitter.com/search?q=${ticker}")
    st.markdown("</div>", unsafe_allow_html=True)

with c_fac:
    if l_an:
        mk_rng = lambda v: f"{v*0.985:.1f}-{v*1.015:.1f}"
        res_rows = "".join([f"<div class='l-item'><span>压力 ({p['d']})</span><span style='color:#C2410C'>{mk_rng(p['v'])}<span class='{'tg-s' if p['l']=='小' else 'tg-m' if p['l']=='中' else 'tg-h'}'>{p['l']}</span></span></div>" for p in l_an['ress']])
        sup_rows = "".join([f"<div class='l-item'><span>支撑 ({p['d']})</span><span style='color:#15803D'>{mk_rng(p['v'])}<span class='{'tg-s' if p['l']=='小' else 'tg-m' if p['l']=='中' else 'tg-h'}'>{p['l']}</span></span></div>" for p in l_an['sups']])
        st.markdown(f"<div class='l-box'><div class='l-title'>🦁 视野·交易计划 ({ticker})</div><div class='l-sub'>增速与估值</div><div class='l-item'><span>未来增速 (Rev)</span><span>{fmt_pct(l_an['growth'])}</span></div><div class='l-item'><span>前瞻合理估值 (25x-35x)</span><span style='font-weight:bold'>{l_an['val_range']}</span></div><div class='l-item'><span>技术面诊断</span><span style='font-weight:bold; color:#2563EB'>{l_an['tech']}</span></div><div class='l-sub'>关键点位 (Support/Resist)</div>{res_rows}{sup_rows}</div>", unsafe_allow_html=True)

if not h.empty:
    st.subheader("🆚 跑赢大盘了吗? (VS SPY/QQQ)")
    cmp = data.get('compare', pd.DataFrame())
    if not cmp.empty:
        alpha_spy = (cmp[ticker].iloc[-1] - cmp['SP500'].iloc[-1]) * 100
        alpha_qqq = (cmp[ticker].iloc[-1] - cmp['Nasdaq'].iloc[-1]) * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=cmp.index, y=cmp[ticker]*100, name=ticker, line=dict(width=3, color='#2563EB')))
        fig2.add_trace(go.Scatter(x=cmp.index, y=cmp['SP500']*100, name="SP500", line=dict(width=1.5, color='gray', dash='dot')))
        fig2.add_trace(go.Scatter(x=cmp.index, y=cmp['Nasdaq']*100, name="Nasdaq", line=dict(width=1.5, color='orange', dash='dot')))
        fig2.update_layout(title=f"Alpha vs SPY: {alpha_spy:+.2f}% | vs QQQ: {alpha_qqq:+.2f}%", height=350, yaxis_title="累计涨幅 (%)", hovermode="x unified", margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📈 核心趋势 (K线+均线+TD九转) [点击展开]", expanded=False):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'], name='K线'))
        fig.add_trace(go.Scatter(x=h.index, y=h['MA20'], line=dict(color='#F59E0B', width=1), name='MA20'))
        fig.add_trace(go.Scatter(x=h.index, y=h['MA60'], line=dict(color='#3B82F6', width=1.5), name='MA60'))
        fig.add_trace(go.Scatter(x=h.index, y=h['MA120'], line=dict(color='#8B5CF6', width=1.5), name='MA120'))
        fig.add_trace(go.Scatter(x=h.index, y=h['MA200'], line=dict(color='#10B981', width=2), name='MA200'))
        
        td_up_mask = h['TD_UP'] > 0; td_down_mask = h['TD_DOWN'] > 0
        if td_up_mask.any(): fig.add_trace(go.Scatter(x=h.index[td_up_mask], y=h.loc[td_up_mask, 'High'] * 1.01, mode="text", text=h.loc[td_up_mask, 'TD_UP'].astype(int), textfont=dict(color='red'), name="TD高点"))
        if td_down_mask.any(): fig.add_trace(go.Scatter(x=h.index[td_down_mask], y=h.loc[td_down_mask, 'Low'] * 0.99, mode="text", text=h.loc[td_down_mask, 'TD_DOWN'].astype(int), textfont=dict(color='green'), name="TD低点"))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='teach-box'><b>🎓 周期拐点教学 (神奇九转)</b>：<br>当 K 线连续 9 天上涨或下跌满足结构时，会出现数字。<b>红色 9</b> 代表上升动能耗尽(潜在卖点)，<b>绿色 9</b> 代表下跌动能衰竭(潜在买点)。</div>", unsafe_allow_html=True)
        
    with st.expander("📅 历史季节性透视 (5年) [点击展开]", expanded=False):
        seas = calculate_seasonality(h)
        if seas is not None:
            fig_seas = make_subplots(specs=[[{"secondary_y": True}]])
            fig_seas.add_trace(go.Bar(x=seas.index, y=seas['Avg Return']*100, name='平均回报', marker_color='#3b82f6'))
            fig_seas.add_trace(go.Scatter(x=seas.index, y=seas['Win Rate']*100, name='胜率', line=dict(color='orange')), secondary_y=True)
            fig_seas.update_layout(title="5年季节性回报统计", height=350, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_seas, use_container_width=True)
            st.markdown("<div class='note-box'><b>📚 读图指南：</b><br><b>蓝色柱 (平均回报)</b>：该股在历史上各个月份的平均涨跌幅。柱子越高，代表该月往往是大肉。<br><b>橙色线 (胜率)</b>：该月份收涨的概率。100% 代表过去5年该月全涨，50% 代表涨跌对半。</div>", unsafe_allow_html=True)
    
    with st.expander("📉 进阶指标 (OBV/MACD/RSI/KDJ/BOLL/筹码) [点击展开]", expanded=False):
        vp_price, vp_vol = calculate_volume_profile(h.iloc[-252:])
        fig3 = make_subplots(rows=5, cols=2, shared_xaxes=True, row_heights=[0.2]*5, column_widths=[0.85, 0.15], horizontal_spacing=0.01, vertical_spacing=0.05, specs=[[{"colspan":2}, None], [{"colspan":2}, None], [{"colspan":2}, None], [{"colspan":2}, None], [{"colspan":1}, {}]])
        fig3.add_trace(go.Scatter(x=h.index, y=h['OBV'], line=dict(color='black', width=1), name='OBV', fill='tozeroy'), row=1, col=1)
        colors = ['red' if v < 0 else 'green' for v in h['Hist']]
        fig3.add_trace(go.Bar(x=h.index, y=h['Hist'], marker_color=colors, name='MACD'), row=2, col=1)
        fig3.add_trace(go.Scatter(x=h.index, y=h['MACD'], line=dict(color='blue'), name='DIF'), row=2, col=1)
        fig3.add_trace(go.Scatter(x=h.index, y=h['Signal'], line=dict(color='orange'), name='DEA'), row=2, col=1)
        fig3.add_trace(go.Scatter(x=h.index, y=h['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
        fig3.add_hline(y=70, line_dash='dot', row=3, col=1); fig3.add_hline(y=30, line_dash='dot', row=3, col=1)
        fig3.add_trace(go.Scatter(x=h.index, y=h['K'], line=dict(color='orange', width=1), name='K'), row=4, col=1)
        fig3.add_trace(go.Scatter(x=h.index, y=h['D'], line=dict(color='blue', width=1), name='D'), row=4, col=1)
        fig3.add_trace(go.Scatter(x=h.index, y=h['J'], line=dict(color='purple', width=1), name='J'), row=4, col=1)
        fig3.add_trace(go.Scatter(x=h.index, y=h['UPPER'], line=dict(color='gray', width=1), name='Upper'), row=5, col=1)
        fig3.add_trace(go.Scatter(x=h.index, y=h['LOWER'], line=dict(color='gray', width=1), name='Lower', fill='tonexty'), row=5, col=1)
        fig3.add_trace(go.Scatter(x=h.index, y=h['Close'], line=dict(color='blue', width=1), name='Close'), row=5, col=1)
        fig3.add_trace(go.Bar(x=vp_vol, y=vp_price, orientation='h', marker_color='rgba(0,0,0,0.3)', name='Vol Profile'), row=5, col=2)
        fig3.update_layout(height=1000, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("<div class='teach-box'><b>🎓 指标大师课</b><br>1. <b>OBV</b>：汽车油门。股价涨OBV没涨=背离(要跌)。<br>2. <b>MACD</b>：红柱多头，绿柱空头。0轴上方金叉最强。<br>3. <b>RSI</b>：>70超买，<30超卖。<br>4. <b>🐳 筹码分布</b>：柱子最长的地方是“筹码密集区”，跌到这里会有强支撑。</div>", unsafe_allow_html=True)

with st.expander("🦁 市场雷达 (做空/分析师/分红) [点击展开]", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("做空比例", fmt_pct(i.get('shortPercentOfFloat')))
    c2.metric("Beta", fmt_num(i.get('beta')))
    c3.metric("回补天数", fmt_num(i.get('shortRatio')))
    c4.metric("股息率", fmt_pct(i.get('dividendYield')))
    st.markdown("<div class='note-box'><b>📖 雷达读数详解：</b><br>🔴 <b>做空比例</b>: >20% 极高风险(但也可能逼空)。<br>🎢 <b>Beta</b>: >1.5 高波动; <0.8 避险。<br>⏳ <b>回补天数</b>: >5天 空头难跑，利多。<br></div>", unsafe_allow_html=True)

bulls, bears = generate_bull_bear_thesis(h, i)
with st.expander("🐂 vs 🐻 智能多空博弈 (AI Thesis) [点击展开]", expanded=True):
    c_bull, c_bear = st.columns(2)
    with c_bull: st.markdown(f"<div class='thesis-col thesis-bull'><b>🚀 多头逻辑 (Bull Case)</b><br>{'<br>'.join([f'✅ {b}' for b in bulls])}</div>", unsafe_allow_html=True)
    with c_bear: st.markdown(f"<div class='thesis-col thesis-bear'><b>🔻 空头逻辑 (Bear Case)</b><br>{'<br>'.join([f'⚠️ {b}' for b in bears])}</div>", unsafe_allow_html=True)

tabs = st.tabs(["📰 资讯/评级", "👥 筹码结构", "💰 估值", "🔮 宏观与期权", "📊 财报"])

with tabs[0]:
    c_n, c_r = st.columns(2)
    with c_n:
        st.subheader("智能新闻")
        news_df = process_news(data['news'])
        if not news_df.empty:
            st.dataframe(news_df[['时间','标题','价格','链接']], column_config={"链接": st.column_config.LinkColumn("阅读"), "价格": st.column_config.TextColumn("🎯", width="small")}, hide_index=True)
        else: st.info("暂无新闻")
    with c_r:
        st.subheader("机构评级")
        if data['upgrades'] is not None:
            u = data['upgrades'].copy()
            u['Firm'] = u['Firm'].apply(lambda x: smart_translate(x, FAMOUS_INSTITUTIONS))
            u['ToGrade'] = u['ToGrade'].apply(lambda x: smart_translate(x, RATING_MAP))
            st.dataframe(u.head(15), use_container_width=True)

with tabs[1]:
    c_ins, c_inr = st.columns(2)
    with c_ins:
        st.subheader("🏦 机构持仓")
        if data['inst'] is not None:
            idf = data['inst'].copy()
            idf = idf.rename(columns={'Holder':'机构', 'pctHeld':'占比', 'Shares':'股数', 'Value':'市值'})
            if '机构' in idf.columns: idf['机构'] = idf['机构'].apply(lambda x: smart_translate(x, FAMOUS_INSTITUTIONS))
            if '占比' in idf.columns: idf['占比'] = idf['占比'].apply(fmt_pct)
            st.dataframe(idf, use_container_width=True)
    with c_inr:
        st.subheader("🕴️ 内部交易")
        if data['insider'] is not None:
            ins_df = data['insider'].copy()
            buys = ins_df[ins_df['Transaction'].str.contains('Buy|Purchase', case=False, na=False)]['Value'].sum()
            sells = ins_df[ins_df['Transaction'].str.contains('Sale', case=False, na=False)]['Value'].sum()
            net = buys - sells
            c_net = "green" if net > 0 else "red"
            st.markdown(f"<b>近6个月净买入/卖出:</b> <span style='color:{c_net};font-size:16px'>${net:,.0f}</span> (买:{fmt_big(buys)} | 卖:{fmt_big(sells)})", unsafe_allow_html=True)
            tgt = ['Insider', 'Relation', 'Start Date', 'Transaction', 'Value', 'Shares']
            ins_df = ins_df[[c for c in tgt if c in ins_df.columns]]
            st.dataframe(ins_df.head(20), use_container_width=True)

with tabs[2]:
    st.subheader("⚖️ 格雷厄姆合理价")
    eps = i.get('trailingEps', 0); bvps = i.get('bookValue', 0)
    if eps > 0 and bvps > 0:
        graham = (22.5 * eps * bvps) ** 0.5
        st.metric("Graham Number", f"${graham:.2f}", f"{(graham-rt_price)/rt_price:.1%} Upside")
    else: st.error("数据不足")
    st.markdown("---")
    st.subheader("💰 DCF 模型")
    peg = i.get('pegRatio')
    if peg:
        peg_color = "green" if peg < 1 else "orange" if peg < 2 else "red"
        st.caption(f"PEG: : {peg} <span style='color:{peg_color}'>●</span> ( <1 低估, >2 高估 )", unsafe_allow_html=True)
    g = st.slider("预期增长率 %", 0, 50, 15)
    if eps > 0:
        val = (eps * ((1+g/100)**5) * 25) / (1.1**5)
        st.metric("估值", f"${val:.2f}")

with tabs[3]:
    c_opt, c_macro = st.columns(2)
    with c_opt:
        st.subheader("🦅 期权异动雷达 (最近期)")
        opt = data.get('options')
        if opt:
            calls = opt['calls']; puts = opt['puts']
            pcr = puts['volume'].sum() / calls['volume'].sum() if calls['volume'].sum() > 0 else 0
            max_pain = calculate_max_pain(calls, puts)
            c_o1, c_o2 = st.columns(2)
            c_o1.metric("Put/Call Ratio", f"{pcr:.2f}", help=">1.0 看空, <0.7 看多")
            c_o2.metric("最大痛点 (Max Pain)", f"${max_pain}", help="机构最希望结算的价位")
            st.caption(f"合约日期: {opt['date']}")
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Bar(x=calls['strike'], y=calls['openInterest'], name='Call OI', marker_color='green'))
            fig_opt.add_trace(go.Bar(x=puts['strike'], y=puts['openInterest'], name='Put OI', marker_color='red'))
            fig_opt.update_layout(title="未平仓合约分布 (Open Interest)", barmode='overlay', height=300)
            st.plotly_chart(fig_opt, use_container_width=True)
        else: st.info("暂无期权数据")
    with c_macro:
        st.subheader("🌍 宏观 & 联动 (实时)")
        macro = fetch_macro_data()
        if macro is not None:
            c_m1, c_m2, c_m3 = st.columns(3)
            vix = macro['^VIX']; tnx = macro['^TNX']; dxy = macro['DX-Y.NYB']
            c_m1.metric("VIX", f"{vix:.2f}")
            c_m2.metric("10年美债", f"{tnx:.2f}%")
            c_m3.metric("美元指数", f"{dxy:.2f}")
            st.markdown("---")
            st.caption(f"{ticker} 与主要资产的 1年 相关性:")
            corrs = fetch_correlation_data(ticker)
            if corrs is not None: st.bar_chart(corrs, height=150)
        else: st.info("宏观数据加载失败")

with tabs[4]:
    if data['fin'] is not None:
        fdf = data['fin'].copy()
        fdf.index = [smart_translate(x, FIN_MAP) for x in fdf.index]
        st.subheader("📊 核心业绩趋势")
        fig_fin = go.Figure()
        if 'Total Revenue' in fdf.columns:
            fig_fin.add_trace(go.Bar(x=fdf.index, y=fdf['Total Revenue'], name='营收', marker_color='#3b82f6'))
        if 'Net Income' in fdf.columns:
            fig_fin.add_trace(go.Bar(x=fdf.index, y=fdf['Net Income'], name='净利润', marker_color='#10b981'))
        fig_fin.update_layout(height=300, hovermode="x unified", barmode='group')
        st.plotly_chart(fig_fin, use_container_width=True)
        with st.expander("查看详细报表"):
            st.dataframe(fdf, use_container_width=True)
    else: st.write("无财报数据")