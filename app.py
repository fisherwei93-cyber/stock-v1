import streamlit as st
import os
import datetime
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re # [核心] 正则表达式模块，防止NameError
import yfinance as yf # [核心] 全局导入

# ================= 1. 铁律配置 (V1.0 Final) =================
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if key in os.environ:
        del os.environ[key]

# 注入自定义图标
ICON_URL = "https://cdn-icons-png.flaticon.com/512/10452/10452449.png"

st.set_page_config(page_title="摩根·V1", layout="wide", page_icon="🦁")

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

    /* 顶部导航显隐 (保留显示，方便操作) */
    header {{ visibility: visible !important; }}

    /* 指标高亮 */
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
    
    /* 折叠栏优化 */
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

    /* 视野黄框 */
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
    
    /* 财报日历样式 */
    .earning-row {{
        display: flex; justify-content: space-between; padding: 8px; 
        border-bottom: 1px solid #333; font-size: 13px;
    }}
    .earning-soon {{ border-left: 3px solid #ef4444; background: rgba(239, 68, 68, 0.1); }}
    
    /* 列表项 */
    .wl-row {{ background-color: #1A1A1A; padding: 12px; margin-bottom: 8px; border-radius: 6px; border-left: 4px solid #555; cursor: pointer; display: flex; justify-content: space-between; align-items: center; border: 1px solid #333; color: #FFFFFF; }}
</style>
""", unsafe_allow_html=True)

# ================= 3. 辅助函数 =================
def fmt_pct(v): return f"{v:.2%}" if isinstance(v, (int, float)) else "-"
def fmt_num(v): return f"{v:.2f}" if isinstance(v, (int, float)) else "-"
def fmt_big(v): 
    if not isinstance(v, (int, float)): return "-"
    if v > 1e12: return f"{v/1e12:.2f}T"
    if v > 1e9: return f"{v/1e9:.2f}B"
    if v > 1e6: return f"{v/1e6:.2f}M"
    return str(v)

if 'watchlist' not in st.session_state: st.session_state.watchlist = ['TSLA', 'NVDA', 'AAPL', 'AMD', 'PLTR']
if 'current_ticker' not in st.session_state: st.session_state.current_ticker = 'TSLA'

# ================= 4. 数据引擎 (Core) =================

# [NEW] 明星股财报日历 (12小时缓存，极少占用资源)
@st.cache_data(ttl=43200, show_spinner=False)
def fetch_star_earnings_calendar():
    stars = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "AMZN", "GOOG", "META", "PLTR", "COIN", "BABA"]
    data = []
    today = datetime.date.today()
    
    for t in stars:
        try:
            s = yf.Ticker(t)
            cal = s.calendar
            e_date = "N/A"
            # 兼容新旧 yfinance 格式
            if isinstance(cal, dict) and cal:
                if 'Earnings Date' in cal: e_date = str(cal['Earnings Date'][0])
                elif 'Earnings High' in cal: e_date = str(cal.get('Earnings Date', [])[0])
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                e_date = cal.iloc[0, 0].strftime("%Y-%m-%d")
            
            if e_date != "N/A":
                ed = datetime.datetime.strptime(str(e_date).split()[0], "%Y-%m-%d").date()
                if ed >= today: # 只显示未来的
                    days_left = (ed - today).days
                    data.append({"Code": t, "Date": str(ed), "Days": days_left})
        except: pass
    
    # 按日期排序
    if data:
        df = pd.DataFrame(data).sort_values("Days")
        return df.to_dict('records')
    return []

# 🔴 快通道：实时价格 (30s缓存)
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
        
        # 盘前盘后逻辑 (仅当有明确数据时显示)
        ext_price, ext_label = None, ""
        try:
            info = s.info if s.info is not None else {}
            pm = info.get('preMarketPrice')
            post = info.get('postMarketPrice')
            if pm and abs(pm - price) > 0.01: ext_price, ext_label = pm, "盘前"
            elif post and abs(post - price) > 0.01: ext_price, ext_label = post, "盘后"
        except: pass

        return {"price": price, "prev": prev, "ext_price": ext_price, "ext_label": ext_label}
    except:
        return {"price": 0, "prev": 0, "ext_price": None, "ext_label": ""}

# 🔵 慢通道：深度数据 (1h缓存)
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_heavy_data(ticker):
    import yfinance as yf
    max_retries = 3
    h = pd.DataFrame()
    s = yf.Ticker(ticker)
    
    for attempt in range(max_retries):
        try:
            h = s.history(period="2y")
            if not h.empty: break
            time.sleep(1)
        except: 
            if attempt == max_retries - 1: return {"history": pd.DataFrame(), "info": {}, "error": "Rate Limit"}
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
    
    # Donchian
    h['DC_Upper'] = h['High'].rolling(20).max(); h['DC_Lower'] = h['Low'].rolling(20).min()

    safe_info = s.info if s.info is not None else {}
    
    return {
        "history": h, "info": safe_info,
        "error": None
    }

# ================= 5. 业务逻辑 (News, etc) =================

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

# ================= 6. 文档 & 主程序 =================

def render_documentation():
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
    <div class='wiki-card'><div class='wiki-title'>9. 唐奇安通道</div><div class='wiki-text'><b>原理：</b> 海龟交易法。<br>突破上轨买，跌破下轨卖。</div></div>
    <div class='wiki-card'><div class='wiki-title'>10. Ichimoku (一目均衡)</div><div class='wiki-text'><b>原理：</b> 云带系统。<br>股价在云上为多，云下为空。</div></div>
    <div class='wiki-card'><div class='wiki-title'>11. ADX (趋势强度)</div><div class='wiki-text'><b>原理：</b> 判断有无趋势。<br>>25: 趋势强劲。<br><20: 震荡市(休息)。</div></div>
    <div class='wiki-card'><div class='wiki-title'>12. HMA (赫尔均线)</div><div class='wiki-text'><b>原理：</b> 零滞后均线，比MA更快。</div></div>
    <div class='wiki-card'><div class='wiki-title'>13. 凯利公式</div><div class='wiki-text'><b>原理：</b> 科学仓位管理。告诉你这把牌该下注多少钱。</div></div>
    <div class='wiki-card'><div class='wiki-title'>14. CCI (顺势指标)</div><div class='wiki-text'><b>原理：</b> 抓极端行情。<br>>100: 超买。<br><-100: 超卖。</div></div>
    """, unsafe_allow_html=True)

def render_main_app():
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
        st.warning(f"深度数据暂时不可用: {heavy['error']}")
        h, i = pd.DataFrame(), {}
    else:
        h, i = heavy['history'], heavy['info']

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

        with st.expander("📈 核心趋势 (K线+SuperTrend+Ichimoku) [点击展开]", expanded=False):
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'], name='K线'))
            st_color = ['#22c55e' if c > l else '#ef4444' for c, l in zip(h['Close'], h['ST_Lower'])]
            fig.add_trace(go.Scatter(x=h.index, y=h['ST_Lower'], mode='markers', marker=dict(color=st_color, size=2), name='SuperTrend'))
            fig.add_trace(go.Scatter(x=h.index, y=h['Senkou_Span_A'], line=dict(color='rgba(0,0,0,0)'), showlegend=False))
            fig.add_trace(go.Scatter(x=h.index, y=h['Senkou_Span_B'], fill='tonexty', fillcolor='rgba(59, 130, 246, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='Ichimoku Cloud'))
            fig.add_trace(go.Scatter(x=h.index, y=h['DC_Upper'], line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'), name='唐奇安上轨'))
            fig.add_trace(go.Scatter(x=h.index, y=h['DC_Lower'], line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'), name='唐奇安下轨'))
            for idx in range(len(h)-50, len(h)): 
                if h['FVG_Bull'].iloc[idx]: fig.add_shape(type="rect", x0=h.index[idx-2], y0=h['Low'].iloc[idx], x1=h.index[idx], y1=h['High'].iloc[idx-2], fillcolor="rgba(139, 92, 246, 0.3)", line_width=0)
            fig.add_trace(go.Scatter(x=h.index, y=h['VWAP'], line=dict(color='#fcd34d', width=2), name='VWAP'))
            fig.add_trace(go.Scatter(x=h.index, y=h['HMA'], line=dict(color='#ec4899', width=1), name='HMA'))
            fig.update_layout(height=800, xaxis_rangeslider_visible=True, margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True, legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div class='teach-box'><b>✨ 新功能</b><br>1. <b>SuperTrend</b>：红绿点趋势。<br>2. <b>云带(Ichimoku)</b>：蓝色阴影区为云，云上做多。<br>3. <b>FVG</b>：紫色缺口。</div>", unsafe_allow_html=True)
            
        with st.expander("📅 历史季节性 & 蒙特卡洛预测 [点击展开]", expanded=False):
            c_seas, c_mc = st.columns(2)
            with c_seas:
                seas = calculate_seasonality(h)
                if seas is not None:
                    fig_seas = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_seas.add_trace(go.Bar(x=seas.index, y=seas['Avg Return']*100, name='平均回报', marker_color='#3b82f6'))
                    fig_seas.add_trace(go.Scatter(x=seas.index, y=seas['Win Rate']*100, name='胜率', line=dict(color='#f97316')), secondary_y=True)
                    fig_seas.update_layout(title="季节性回报", height=350, margin=dict(l=0,r=0,t=30,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_seas, use_container_width=True)
            with c_mc:
                last_price = h['Close'].iloc[-1]; daily_vol = h['Close'].pct_change().std()
                simulations = 100; days = 30; sim_df = pd.DataFrame()
                for x in range(simulations):
                    price_series = [last_price]
                    for y in range(days): price_series.append(price_series[-1] * (1 + np.random.normal(0, daily_vol)))
                    sim_df[x] = price_series
                fig_mc = go.Figure()
                for col in sim_df.columns: fig_mc.add_trace(go.Scatter(y=sim_df[col], mode='lines', line=dict(color='rgba(59, 130, 246, 0.1)', width=1), showlegend=False))
                fig_mc.add_trace(go.Scatter(y=[last_price]*days, mode='lines', line=dict(color='red', dash='dash'), name='当前价'))
                fig_mc.update_layout(title=f"未来30天价格模拟 ({simulations}次)", height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_mc, use_container_width=True)
                final_prices = sim_df.iloc[-1].values
                p5 = np.percentile(final_prices, 5); p50 = np.percentile(final_prices, 50); p95 = np.percentile(final_prices, 95)
                st.markdown(f"<div class='mc-box'><span style='color:#fca5a5'>📉 底线(P5): <b>${p5:.2f}</b></span> <span style='color:#86efac'>🚀 乐观(P95): <b>${p95:.2f}</b></span></div>", unsafe_allow_html=True)
        
        with st.expander("📉 进阶指标 (Z-Score/ADX/CCI/WR) [点击展开]", expanded=False):
            vp_price, vp_vol = calculate_volume_profile(h.iloc[-252:])
            fig3 = make_subplots(rows=7, cols=2, shared_xaxes=True, row_heights=[0.14]*7, column_widths=[0.85, 0.15], horizontal_spacing=0.01, vertical_spacing=0.03, specs=[[{"colspan":2}, None], [{"colspan":2}, None], [{"colspan":2}, None], [{"colspan":2}, None], [{"colspan":2}, None], [{"colspan":2}, None], [{"colspan":1}, {}]])
            
            # Z-Score
            fig3.add_trace(go.Scatter(x=h.index, y=h['Z_Score'], line=dict(color='#f472b6', width=1), name='Z-Score'), row=1, col=1)
            fig3.add_hline(y=2, line_dash='dot', row=1, col=1); fig3.add_hline(y=-2, line_dash='dot', row=1, col=1)
            # ADX
            fig3.add_trace(go.Scatter(x=h.index, y=h['ADX'], line=dict(color='#fbbf24', width=1), name='ADX (强度)'), row=2, col=1)
            fig3.add_hline(y=25, line_dash='dot', row=2, col=1)
            # CCI
            fig3.add_trace(go.Scatter(x=h.index, y=h['CCI'], line=dict(color='#22d3ee', width=1), name='CCI'), row=3, col=1)
            fig3.add_hline(y=100, line_dash='dot', row=3, col=1); fig3.add_hline(y=-100, line_dash='dot', row=3, col=1)
            # CMF
            cmf_col = ['#22c55e' if v >= 0 else '#ef4444' for v in h['CMF']]
            fig3.add_trace(go.Bar(x=h.index, y=h['CMF'], marker_color=cmf_col, name='CMF资金'), row=4, col=1)
            # WR
            fig3.add_trace(go.Scatter(x=h.index, y=h['WR'], line=dict(color='#06b6d4', width=1), name='Williams %R'), row=5, col=1)
            fig3.add_hline(y=-20, line_dash='dot', row=5, col=1); fig3.add_hline(y=-80, line_dash='dot', row=5, col=1)
            # MACD
            colors = ['#ef4444' if v < 0 else '#22c55e' for v in h['Hist']]
            fig3.add_trace(go.Bar(x=h.index, y=h['Hist'], marker_color=colors, name='MACD'), row=6, col=1)
            fig3.add_trace(go.Scatter(x=h.index, y=h['MACD'], line=dict(color='#3b82f6'), name='DIF'), row=6, col=1)
            fig3.add_trace(go.Scatter(x=h.index, y=h['Signal'], line=dict(color='#f97316'), name='DEA'), row=6, col=1)
            # BOLL + Profile
            fig3.add_trace(go.Scatter(x=h.index, y=h['UPPER'], line=dict(color='#6b7280', width=1), name='Upper'), row=7, col=1)
            fig3.add_trace(go.Scatter(x=h.index, y=h['LOWER'], line=dict(color='#6b7280', width=1), name='Lower', fill='tonexty'), row=7, col=1)
            fig3.add_trace(go.Scatter(x=h.index, y=h['Close'], line=dict(color='#3b82f6', width=1), name='Close'), row=7, col=1)
            fig3.add_trace(go.Bar(x=vp_vol, y=vp_price, orientation='h', marker_color='rgba(100,100,100,0.3)', name='Vol Profile'), row=7, col=2)
            
            fig3.update_layout(height=1400, margin=dict(l=0,r=0,t=10,b=0), showlegend=True, legend=dict(orientation="h", y=1.01), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig3, use_container_width=True)

    # 核心数据 & 财报日历
    st.subheader("📊 核心数据")
    c1, c2, c3 = st.columns(3)
    safe_i = i if isinstance(i, dict) else {}
    c1.metric("市值", fmt_big(safe_i.get('marketCap')))
    c2.metric("做空比", fmt_pct(safe_i.get('shortPercentOfFloat')))
    c3.metric("股息率", fmt_pct(safe_i.get('dividendYield')))
    
    st.markdown("---")
    st.markdown("<div class='note-box'><b>📖 雷达读数详解：</b><br>🔴 <b>做空比例</b>: >20% 极高风险。<br>🎢 <b>Beta</b>: >1.5 高波动。<br>⏳ <b>回补天数</b>: >5天 空头难跑。</div>", unsafe_allow_html=True)

    tabs = st.tabs(["📰 资讯/评级", "👥 筹码/内部人", "💰 估值", "🔮 宏观与期权", "📊 财报", "🎓 深度研报"])

    with tabs[0]:
        c_n, c_r = st.columns(2)
        with c_n:
            st.subheader("智能新闻")
            news_df = process_news(heavy.get('news', [])) # Safe get
            if not news_df.empty: st.dataframe(news_df[['时间','标题','价格','链接']], column_config={"链接": st.column_config.LinkColumn("阅读"), "价格": st.column_config.TextColumn("🎯", width="small")}, hide_index=True)
            else: st.info("暂无新闻")
        with c_r:
            st.subheader("机构评级")
            if heavy.get('upgrades') is not None:
                u = heavy['upgrades'].copy()
                u['Firm'] = u['Firm'].apply(lambda x: smart_translate(x, FAMOUS_INSTITUTIONS))
                u['ToGrade'] = u['ToGrade'].apply(lambda x: smart_translate(x, RATING_MAP))
                st.dataframe(u.head(15), use_container_width=True)

    with tabs[1]:
        c_ins, c_inr = st.columns(2)
        with c_ins:
            st.subheader("🏦 机构持仓")
            if heavy.get('inst') is not None:
                idf = heavy['inst'].copy()
                idf = idf.rename(columns={'Holder':'机构', 'pctHeld':'占比', 'Shares':'股数', 'Value':'市值'})
                if '机构' in idf.columns: idf['机构'] = idf['机构'].apply(lambda x: smart_translate(x, FAMOUS_INSTITUTIONS))
                if '占比' in idf.columns: idf['占比'] = idf['占比'].apply(fmt_pct)
                st.dataframe(idf, use_container_width=True)
        with c_inr:
            st.subheader("🕴️ 内部交易")
            if heavy.get('insider') is not None and not heavy['insider'].empty:
                ins_df = heavy['insider'].copy()
                try:
                    ins_df['Date'] = pd.to_datetime(ins_df['Start Date'])
                    ins_df['Type'] = ins_df['Transaction'].apply(lambda x: 'Buy' if 'Buy' in str(x) or 'Purchase' in str(x) else 'Sell' if 'Sale' in str(x) else 'Other')
                    ins_df = ins_df[ins_df['Type'].isin(['Buy','Sell'])]
                    fig_ins = px.scatter(ins_df, x='Date', y='Value', size='Shares', color='Type', color_discrete_map={'Buy':'#4ade80', 'Sell':'#f87171'}, hover_data=['Insider'])
                    fig_ins.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_ins, use_container_width=True)
                    st.dataframe(ins_df[['Insider', 'Date', 'Transaction', 'Value']].head(10), use_container_width=True)
                except: st.warning("数据格式暂不支持可视化")
            else: st.info("📊 暂无内部人交易数据")

    with tabs[2]:
        st.subheader("⚖️ 格雷厄姆合理价")
        eps = safe_i.get('trailingEps', 0); bvps = safe_i.get('bookValue', 0)
        if eps is not None and bvps is not None and eps > 0 and bvps > 0:
            graham = (22.5 * eps * bvps) ** 0.5
            st.metric("Graham Number", f"${graham:.2f}", f"{(graham-rt_price)/rt_price:.1%} Upside")
        else: st.error("数据不足 (EPS/BVPS缺失)")
        st.markdown("---")
        st.subheader("💰 DCF 模型")
        peg = safe_i.get('pegRatio')
        if peg:
            peg_color = "#4ade80" if peg < 1 else "#fbbf24" if peg < 2 else "#f87171"
            st.caption(f"PEG: : {peg} <span style='color:{peg_color}'>●</span>", unsafe_allow_html=True)
        g = st.slider("预期增长率 %", 0, 50, 15)
        if eps is not None and eps > 0:
            val = (eps * ((1+g/100)**5) * 25) / (1.1**5)
            st.metric("估值", f"${val:.2f}")

    with tabs[3]:
        c_opt, c_macro = st.columns(2)
        with c_opt:
            st.subheader("🦅 期权异动")
            opt = heavy.get('options')
            if opt:
                calls = opt['calls']; puts = opt['puts']
                pcr = puts['volume'].sum() / calls['volume'].sum() if calls['volume'].sum() > 0 else 0
                max_pain = calculate_max_pain(calls, puts)
                c_o1, c_o2 = st.columns(2)
                c_o1.metric("Put/Call Ratio", f"{pcr:.2f}")
                c_o2.metric("最大痛点", f"${max_pain}")
                fig_opt = go.Figure()
                fig_opt.add_trace(go.Bar(x=calls['strike'], y=calls['openInterest'], name='Call OI', marker_color='green'))
                fig_opt.add_trace(go.Bar(x=puts['strike'], y=puts['openInterest'], name='Put OI', marker_color='red'))
                fig_opt.update_layout(barmode='overlay', height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_opt, use_container_width=True)
            else: st.info("暂无期权数据")
        with c_macro:
            st.subheader("🌍 宏观联动")
            corrs = fetch_correlation_data(ticker)
            if corrs is not None: st.bar_chart(corrs, height=150)
            else: st.info("宏观数据加载失败")

    with tabs[4]:
        if heavy.get('fin') is not None:
            fdf = heavy['fin'].copy()
            fdf.index = [smart_translate(x, FIN_MAP) for x in fdf.index]
            st.subheader("📊 业绩趋势")
            fig_fin = go.Figure()
            if 'Total Revenue' in fdf.columns: fig_fin.add_trace(go.Bar(x=fdf.index, y=fdf['Total Revenue'], name='营收', marker_color='#3b82f6'))
            if 'Net Income' in fdf.columns: fig_fin.add_trace(go.Bar(x=fdf.index, y=fdf['Net Income'], name='净利润', marker_color='#10b981'))
            fig_fin.update_layout(height=300, hovermode="x unified", barmode='group', template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_fin, use_container_width=True)
            with st.expander("查看详细报表"): st.dataframe(fdf, use_container_width=True)
        else: st.write("无财报数据")
        
    with tabs[5]: 
        st.header(f"🎓 {ticker} 深度研报")
        st.markdown("<div class='report-title'>1. 🏢 商业模式</div>", unsafe_allow_html=True)
        summary = safe_i.get('longBusinessSummary', '暂无描述')
        st.markdown(f"<div class='report-text'>{summary}</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='report-title'>2. 🏰 护城河分析</div>", unsafe_allow_html=True)
        gross_margin = safe_i.get('grossMargins', 0)
        roe = safe_i.get('returnOnEquity', 0)
        gm_color = "#4ade80" if gross_margin and gross_margin > 0.4 else "#f87171"
        roe_color = "#4ade80" if roe and roe > 0.15 else "#f87171"
        c_m1, c_m2 = st.columns(2)
        c_m1.markdown(f"<div class='score-card'><div class='sc-lbl'>毛利率</div><div class='sc-val' style='color:{gm_color}'>{fmt_pct(gross_margin)}</div><div class='sc-lbl'>巴菲特标准: >40%</div></div>", unsafe_allow_html=True)
        c_m2.markdown(f"<div class='score-card'><div class='sc-lbl'>ROE</div><div class='sc-val' style='color:{roe_color}'>{fmt_pct(roe)}</div><div class='sc-lbl'>巴菲特标准: >15%</div></div>", unsafe_allow_html=True)
        
        st.markdown("<div class='report-title'>3. 🧘‍♂️ 大师清单</div>", unsafe_allow_html=True)
        peg = safe_i.get('pegRatio')
        lynch_pass = peg is not None and peg < 1.0
        st.markdown(f"<div class='guru-check'><span style='font-size:20px; margin-right:10px'>{'✅' if lynch_pass else '❌'}</span><div><b>彼得·林奇法则</b><br><span style='color:#9ca3af; font-size:13px'>PEG < 1.0 (当前: {peg})</span></div></div>", unsafe_allow_html=True)
        
        graham_pass = False
        if eps is not None and bvps is not None and eps > 0 and bvps > 0:
            graham_price = (22.5 * eps * bvps) ** 0.5
            graham_pass = rt_price < graham_price
            st.markdown(f"<div class='guru-check'><span style='font-size:20px; margin-right:10px'>{'✅' if graham_pass else '❌'}</span><div><b>格雷厄姆法则</b><br><span style='color:#9ca3af; font-size:13px'>股价 < ${graham_price:.2f}</span></div></div>", unsafe_allow_html=True)
            
        st.markdown("<div class='report-title'>4. 📞 尽职调查</div>", unsafe_allow_html=True)
        dd_c1, dd_c2, dd_c3 = st.columns(3)
        dd_c1.link_button("📄 SEC 10-K", f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}")
        dd_c2.link_button("🗣️ Earnings Call", f"https://www.google.com/search?q={ticker}+earnings+call+transcript")
        dd_c3.link_button("🌐 Investor Relations", f"https://www.google.com/search?q={ticker}+investor+relations")

# ================= 6. 执行逻辑 =================
# 左侧栏
with st.sidebar:
    st.title("🦁 摩根·V1")
    new_ticker = st.text_input("🔍 搜索 (如 AAPL)", "").upper()
    if new_ticker:
        st.session_state.current_ticker = new_ticker
        st.rerun()

    # [NEW] 明星股财报日历
    st.markdown("---")
    st.caption("📅 明星股财报日历")
    star_cal = fetch_star_earnings_calendar()
    if star_cal:
        for row in star_cal:
            # 只有7天内的才高亮
            bg_style = "earning-soon" if row['Days'] >= 0 and row['Days'] <= 7 else ""
            st.markdown(f"""
            <div class='earning-row {bg_style}'>
                <span style='font-weight:bold; color:#fff'>{row['Code']}</span>
                <span style='color:#9ca3af'>{row['Date']} (T-{row['Days']})</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("暂无近期数据")

    st.markdown("---")
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

# 页面路由
page = st.sidebar.radio("📌 导航", ["🚀 股票分析", "📖 功能说明书"])

if page == "🚀 股票分析":
    render_main_app()
else:
    render_documentation()