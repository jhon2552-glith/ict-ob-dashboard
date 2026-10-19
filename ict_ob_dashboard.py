import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="ICT Order Block Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark + mobile friendly
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #252b3b 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #2d3548;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00d4aa;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b95a8;
        margin-top: 4px;
    }
    h1, h2, h3 {
        color: #f0f2f6 !important;
    }
    .stSelectbox label, .stSlider label, .stRadio label {
        color: #c9d1d9 !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #161b22;
    }
    .positive { color: #00d4aa; }
    .negative { color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def get_data(symbol: str, interval: str, period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance"""
    ticker_map = {
        "XAUUSD": "GC=F",
        "NASDAQ": "NQ=F"
    }
    ticker = ticker_map.get(symbol, symbol)
    
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        
        # Handle MultiIndex columns (common in newer yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.dropna()
        
        # Standardize column names
        df.columns = [str(c).title() for c in df.columns]
        
        # Ensure required columns exist
        required = ["Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required):
            st.error(f"Kolom data tidak lengkap: {df.columns.tolist()}")
            return pd.DataFrame()
            
        return df
    except Exception as e:
        st.error(f"Gagal ambil data: {e}")
        return pd.DataFrame()


def find_swing_highs_lows(df: pd.DataFrame, lookback: int = 5):
    """Find swing highs and lows"""
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n - lookback):
        if highs[i] == max(highs[i-lookback:i+lookback+1]):
            swing_high[i] = True
        if lows[i] == min(lows[i-lookback:i+lookback+1]):
            swing_low[i] = True
            
    return swing_high, swing_low


def detect_order_blocks(df: pd.DataFrame, lookback: int = 5, displacement_factor: float = 1.5):
    """
    Detect ICT-style Order Blocks (simplified):
    - Bullish OB: last down-close candle before a strong upward displacement that breaks structure
    - Bearish OB: last up-close candle before a strong downward displacement
    """
    df = df.copy()
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(df)
    
    atr = pd.Series(highs - lows).rolling(14).mean().bfill().fillna(0).values
    
    bullish_ob = []  # list of (index, top, bottom)
    bearish_ob = []
    
    swing_high, swing_low = find_swing_highs_lows(df, lookback)
    
    for i in range(lookback + 2, n - 2):
        # Look for displacement (strong move)
        body = abs(closes[i] - opens[i])
        range_ = highs[i] - lows[i]
        
        # Bullish displacement: strong green candle that breaks recent swing high
        if closes[i] > opens[i] and body > atr[i] * displacement_factor * 0.6:
            # Check if broke a recent swing high
            recent_swings = np.where(swing_high[max(0, i-20):i])[0]
            if len(recent_swings) > 0:
                last_swing_idx = max(0, i-20) + recent_swings[-1]
                if closes[i] > highs[last_swing_idx]:
                    # Find last down-close candle before this move
                    for j in range(i-1, max(0, i-8), -1):
                        if closes[j] < opens[j]:
                            top = max(opens[j], closes[j])
                            bottom = min(opens[j], closes[j])
                            # Prefer body, but include some wick
                            bullish_ob.append({
                                "idx": j,
                                "time": df.index[j],
                                "top": float(highs[j]),
                                "bottom": float(lows[j]),
                                "body_top": float(top),
                                "body_bottom": float(bottom),
                                "type": "bullish",
                                "mitigated": False
                            })
                            break
        
        # Bearish displacement
        if closes[i] < opens[i] and body > atr[i] * displacement_factor * 0.6:
            recent_swings = np.where(swing_low[max(0, i-20):i])[0]
            if len(recent_swings) > 0:
                last_swing_idx = max(0, i-20) + recent_swings[-1]
                if closes[i] < lows[last_swing_idx]:
                    for j in range(i-1, max(0, i-8), -1):
                        if closes[j] > opens[j]:
                            top = max(opens[j], closes[j])
                            bottom = min(opens[j], closes[j])
                            bearish_ob.append({
                                "idx": j,
                                "time": df.index[j],
                                "top": float(highs[j]),
                                "bottom": float(lows[j]),
                                "body_top": float(top),
                                "body_bottom": float(bottom),
                                "type": "bearish",
                                "mitigated": False
                            })
                            break
    
    return bullish_ob, bearish_ob


def get_dealing_range(df: pd.DataFrame, lookback_bars: int = 50):
    """Calculate recent dealing range and equilibrium"""
    recent = df.iloc[-lookback_bars:]
    high = recent["High"].max()
    low = recent["Low"].min()
    eq = (high + low) / 2
    return high, low, eq


def run_backtest(df: pd.DataFrame, bullish_obs, bearish_obs, 
                 risk_reward: float = 2.0, lookback_range: int = 50):
    """
    Simple backtest:
    - Long: Bullish OB in Discount zone, entry on retest, SL below OB, TP = RR
    - Short: Bearish OB in Premium zone
    """
    trades = []
    equity = [10000.0]
    position = None
    
    high_range, low_range, eq = get_dealing_range(df, lookback_range)
    
    # Sort OBs by time
    all_obs = sorted(bullish_obs + bearish_obs, key=lambda x: x["idx"])
    
    for i in range(len(df)):
        current_price = df["Close"].iloc[i]
        current_high = df["High"].iloc[i]
        current_low = df["Low"].iloc[i]
        
        # Update dealing range every 20 bars
        if i % 20 == 0 and i > lookback_range:
            high_range, low_range, eq = get_dealing_range(df.iloc[:i+1], lookback_range)
        
        # Check open position
        if position is not None:
            if position["type"] == "long":
                if current_low <= position["sl"]:
                    pnl = (position["sl"] - position["entry"]) / position["entry"] * 100
                    trades.append({**position, "exit": position["sl"], "exit_idx": i, "pnl_pct": pnl, "result": "loss"})
                    equity.append(equity[-1] * (1 + pnl/100))
                    position = None
                elif current_high >= position["tp"]:
                    pnl = (position["tp"] - position["entry"]) / position["entry"] * 100
                    trades.append({**position, "exit": position["tp"], "exit_idx": i, "pnl_pct": pnl, "result": "win"})
                    equity.append(equity[-1] * (1 + pnl/100))
                    position = None
            else:  # short
                if current_high >= position["sl"]:
                    pnl = (position["entry"] - position["sl"]) / position["entry"] * 100
                    trades.append({**position, "exit": position["sl"], "exit_idx": i, "pnl_pct": pnl, "result": "loss"})
                    equity.append(equity[-1] * (1 + pnl/100))
                    position = None
                elif current_low <= position["tp"]:
                    pnl = (position["entry"] - position["tp"]) / position["entry"] * 100
                    trades.append({**position, "exit": position["tp"], "exit_idx": i, "pnl_pct": pnl, "result": "win"})
                    equity.append(equity[-1] * (1 + pnl/100))
                    position = None
        
        if position is not None:
            continue
            
        # Look for new entries
        for ob in all_obs:
            if ob["idx"] >= i or ob.get("used", False):
                continue
            if i - ob["idx"] > 30:  # OB too old
                continue
                
            # Bullish OB in Discount
            if ob["type"] == "bullish" and current_price < eq:
                # Price retesting the OB zone
                if current_low <= ob["top"] and current_high >= ob["bottom"]:
                    entry = (ob["top"] + ob["bottom"]) / 2
                    sl = ob["bottom"] - (ob["top"] - ob["bottom"]) * 0.2
                    risk = entry - sl
                    tp = entry + risk * risk_reward
                    
                    position = {
                        "type": "long",
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "entry_idx": i,
                        "ob_time": ob["time"],
                        "zone": "Discount"
                    }
                    ob["used"] = True
                    break
            
            # Bearish OB in Premium
            if ob["type"] == "bearish" and current_price > eq:
                if current_high >= ob["bottom"] and current_low <= ob["top"]:
                    entry = (ob["top"] + ob["bottom"]) / 2
                    sl = ob["top"] + (ob["top"] - ob["bottom"]) * 0.2
                    risk = sl - entry
                    tp = entry - risk * risk_reward
                    
                    position = {
                        "type": "short",
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "entry_idx": i,
                        "ob_time": ob["time"],
                        "zone": "Premium"
                    }
                    ob["used"] = True
                    break
    
    return trades, equity


def calculate_metrics(trades, equity):
    if not trades:
        return {
            "total_trades": 0, "winrate": 0, "total_return": 0,
            "avg_rr": 0, "max_dd": 0, "profit_factor": 0
        }
    
    wins = [t for t in trades if t["result"] == "win"]
    losses = [t for t in trades if t["result"] == "loss"]
    
    total_return = (equity[-1] / equity[0] - 1) * 100 if equity else 0
    winrate = len(wins) / len(trades) * 100 if trades else 0
    
    gross_profit = sum(t["pnl_pct"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_pct"] for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Max drawdown
    eq_arr = np.array(equity)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    max_dd = abs(dd.min()) if len(dd) > 0 else 0
    
    return {
        "total_trades": len(trades),
        "winrate": round(winrate, 1),
        "total_return": round(total_return, 2),
        "max_dd": round(max_dd, 2),
        "profit_factor": round(profit_factor, 2),
        "wins": len(wins),
        "losses": len(losses)
    }


# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("⚙️ Settings")
    
    symbol = st.selectbox("Instrument", ["XAUUSD", "NASDAQ"], index=0)
    
    interval = st.selectbox(
        "Timeframe",
        ["15m", "30m", "1h", "4h", "1d"],
        index=2,
        help="Multi-timeframe: pilih salah satu untuk dianalisis"
    )
    
    period = st.selectbox(
        "Period Data",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=2
    )
    
    st.markdown("---")
    st.subheader("Strategy Params")
    
    lookback = st.slider("Swing Lookback", 3, 10, 5)
    disp_factor = st.slider("Displacement Factor", 0.8, 2.5, 1.3, 0.1)
    risk_reward = st.slider("Risk : Reward", 1.0, 4.0, 2.0, 0.5)
    range_lookback = st.slider("Dealing Range Bars", 30, 100, 50)
    
    st.markdown("---")
    run_btn = st.button("🚀 Run Backtest", use_container_width=True, type="primary")

# ==================== MAIN ====================
st.title("📊 ICT Order Block + PD Zone Backtester")
st.caption("XAUUSD & NASDAQ | Simplified ICT Order Block in Premium / Discount")

if run_btn or "data" not in st.session_state:
    with st.spinner("Mengambil data & menganalisa Order Block..."):
        df = get_data(symbol, interval, period)
        
        if df.empty or len(df) < 50:
            st.error("Data tidak cukup. Coba timeframe lebih besar atau period lebih panjang.")
            st.stop()
        
        try:
            bullish_obs, bearish_obs = detect_order_blocks(df, lookback, disp_factor)
            trades, equity = run_backtest(df, bullish_obs, bearish_obs, risk_reward, range_lookback)
            metrics = calculate_metrics(trades, equity)
            high_r, low_r, eq = get_dealing_range(df, range_lookback)
            
            st.session_state.data = df
            st.session_state.bullish_obs = bullish_obs
            st.session_state.bearish_obs = bearish_obs
            st.session_state.trades = trades
            st.session_state.equity = equity
            st.session_state.metrics = metrics
            st.session_state.eq = eq
            st.session_state.high_r = high_r
            st.session_state.low_r = low_r
            st.session_state.symbol = symbol
            st.session_state.interval = interval
        except Exception as e:
            st.error(f"Error saat analisa: {e}")
            st.stop()

if "data" not in st.session_state:
    st.info("Tekan tombol **Run Backtest** di sidebar untuk memulai.")
    st.stop()

df = st.session_state.data
bullish_obs = st.session_state.bullish_obs
bearish_obs = st.session_state.bearish_obs
trades = st.session_state.trades
equity = st.session_state.equity
metrics = st.session_state.metrics
eq = st.session_state.eq
high_r = st.session_state.high_r
low_r = st.session_state.low_r

# ===== METRICS =====
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['total_trades']}</div>
        <div class="metric-label">Total Trades</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    color = "positive" if metrics["winrate"] >= 50 else "negative"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value {color}">{metrics['winrate']}%</div>
        <div class="metric-label">Winrate</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    color = "positive" if metrics["total_return"] >= 0 else "negative"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value {color}">{metrics['total_return']}%</div>
        <div class="metric-label">Total Return</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value negative">{metrics['max_dd']}%</div>
        <div class="metric-label">Max Drawdown</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['profit_factor']}</div>
        <div class="metric-label">Profit Factor</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ===== CHART =====
st.subheader(f"📈 {st.session_state.symbol} — {st.session_state.interval}")

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.7, 0.3],
    subplot_titles=("Price + Order Blocks + PD Zone", "Equity Curve")
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
        increasing_line_color="#00d4aa",
        decreasing_line_color="#ff6b6b"
    ),
    row=1, col=1
)

# Equilibrium line
fig.add_hline(y=eq, line_dash="dash", line_color="#f0c040", 
              annotation_text="Equilibrium", annotation_position="right", row=1, col=1)

# Premium / Discount zones (background)
fig.add_hrect(y0=eq, y1=high_r, fillcolor="rgba(255,107,107,0.08)", 
              line_width=0, row=1, col=1)
fig.add_hrect(y0=low_r, y1=eq, fillcolor="rgba(0,212,170,0.08)", 
              line_width=0, row=1, col=1)

# Plot recent Order Blocks (last 15)
for ob in bullish_obs[-15:]:
    fig.add_shape(
        type="rect",
        x0=ob["time"], x1=df.index[min(ob["idx"]+15, len(df)-1)],
        y0=ob["bottom"], y1=ob["top"],
        fillcolor="rgba(0,212,170,0.25)",
        line=dict(color="#00d4aa", width=1),
        row=1, col=1
    )

for ob in bearish_obs[-15:]:
    fig.add_shape(
        type="rect",
        x0=ob["time"], x1=df.index[min(ob["idx"]+15, len(df)-1)],
        y0=ob["bottom"], y1=ob["top"],
        fillcolor="rgba(255,107,107,0.25)",
        line=dict(color="#ff6b6b", width=1),
        row=1, col=1
    )

# Mark trades
for t in trades:
    color = "#00d4aa" if t["result"] == "win" else "#ff6b6b"
    marker = "triangle-up" if t["type"] == "long" else "triangle-down"
    fig.add_trace(
        go.Scatter(
            x=[df.index[t["entry_idx"]]],
            y=[t["entry"]],
            mode="markers",
            marker=dict(symbol=marker, size=12, color=color, line=dict(width=1, color="white")),
            name=f"{t['type'].upper()} {t['result']}",
            showlegend=False
        ),
        row=1, col=1
    )

# Equity curve
eq_x = list(range(len(equity)))
fig.add_trace(
    go.Scatter(
        x=eq_x,
        y=equity,
        mode="lines",
        line=dict(color="#00d4aa", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.15)",
        name="Equity"
    ),
    row=2, col=1
)

fig.update_layout(
    template="plotly_dark",
    height=700,
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="#c9d1d9")
)

fig.update_xaxes(gridcolor="#1f2937", row=1, col=1)
fig.update_yaxes(gridcolor="#1f2937", row=1, col=1)
fig.update_xaxes(gridcolor="#1f2937", row=2, col=1)
fig.update_yaxes(gridcolor="#1f2937", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ===== TRADE LIST =====
if trades:
    st.subheader("📋 Trade History")
    
    trade_df = pd.DataFrame([
        {
            "Type": t["type"].upper(),
            "Zone": t["zone"],
            "Entry": round(t["entry"], 2),
            "SL": round(t["sl"], 2),
            "TP": round(t["tp"], 2),
            "PnL %": round(t["pnl_pct"], 2),
            "Result": t["result"].upper()
        }
        for t in trades
    ])
    
    def color_result(val):
        if val == "WIN":
            return "color: #00d4aa"
        elif val == "LOSS":
            return "color: #ff6b6b"
        return ""
    
    st.dataframe(
        trade_df.style.map(color_result, subset=["Result"]),
        use_container_width=True,
        height=300
    )
else:
    st.info("Belum ada trade yang ter-trigger dengan parameter saat ini. Coba longgarkan Displacement Factor atau ganti timeframe.")

# ===== INFO =====
with st.expander("ℹ️ Cara Kerja Strategy (Simplified ICT)"):
    st.markdown("""
    **Order Block Detection:**
    - **Bullish OB**: Candle down-close terakhir sebelum displacement naik yang break structure
    - **Bearish OB**: Candle up-close terakhir sebelum displacement turun yang break structure
    
    **Premium / Discount:**
    - Dealing Range diambil dari swing high–low terbaru
    - **Equilibrium** = 50% range
    - Di atas EQ = **Premium** → cari Bearish OB
    - Di bawah EQ = **Discount** → cari Bullish OB
    
    **Entry Rules:**
    - Long: Harga retest Bullish OB yang berada di Discount
    - Short: Harga retest Bearish OB yang berada di Premium
    - SL di luar zona OB, TP sesuai Risk:Reward yang dipilih
    
    > Catatan: Ini adalah versi **simplified**. ICT full (FVG, Breaker, Mitigation, SMT, dll) jauh lebih kompleks.
    """)

st.caption("Dashboard by Grok • Data via Yahoo Finance (GC=F & NQ=F) • Educational purpose only")
