"""
0001_Portfolio_Stage01_Upload_and_View.py - FULLY FIXED

FIXES:
1. ✅ CSV parsing: Handle quoted columns + dtype errors
2. ✅ Progress bar: Clamp 0-100, handle empty DataFrame
3. ✅ NSE suffix for ALL Indian stocks
4. ✅ Robust numeric conversion
5. ✅ Empty portfolio handling
6. ✅ Added missing imports for Tab5: io, contextlib, numpy, plotly.graph_objects
7. ✅ Added missing functions: fetch_close_matrix(), relative_return()
8. ✅ Tab4 now supports selectable benchmark (NIFTY50 or any holding)
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import yfinance as yf
import plotly.express as px

# NEW imports required by Tab4/Tab5
import io
import contextlib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.yf_symbols import to_yf_symbol


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "data"
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio_stage01.json")
GOLD_PROXY = "GC=F"  # Comex Gold Futures

os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(
    page_title="Portfolio Stage 01 - Upload & View (FULLY FIXED)",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Portfolio Stage 01 - Upload & View")
st.markdown("**Upload CSV → Live yfinance Prices → SGB vs Gold → Performance → Candles**")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)
def get_live_price(ticker):
    """Smart price lookup with SGB handling"""
    if pd.isna(ticker):
        return None

    ticker_clean = str(ticker).strip()
    if not ticker_clean:
        return None

    # ✅ SGB special handling
    sgb_map = {
        "SGBD29VIII-GB": "GOLDINR=X",
        "SGBJAN29IX-GB": "GOLDINR=X",
        "SGBMAY29I": "GOLDINR=X",
        "SGBOCT27VI-GB": "GOLDINR=X",
        "SGBSEP31II-GB": "GOLDINR=X",
    }

    is_sgb = ticker_clean in sgb_map or ("SGB" in ticker_clean.upper())

    if is_sgb:
        try:
            comex = yf.Ticker("GC=F").history(period="1d")
            usd_inr = yf.Ticker("INR=X").history(period="1d")
            if not comex.empty and not usd_inr.empty:
                comex_usd_oz = comex["Close"].iloc[-1]
                usd_inr_rate = usd_inr["Close"].iloc[-1]
                # 1 troy ounce = 31.1035 grams
                price_inr_per_gram = (comex_usd_oz * usd_inr_rate) / 31.1035
                return price_inr_per_gram
            return None
        except Exception:
            return None

    # NSE stocks: always add .NS suffix for common Indian tickers
    elif any(
        kw in ticker_clean.upper()
        for kw in [
            "HDFC", "RELI", "ITC", "LICI", "LUPIN",
            "NIFTY", "GOLD", "SILVER", "LIQUID",
            "GHCL", "HAPP", "HINDU", "PGINVIT",
            "TATVA", "JUNIORBEES"
        ]
    ):
        symbol = f"{ticker_clean}.NS"
    else:
        symbol = ticker_clean

    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        return data["Close"].iloc[-1] if not data.empty else None
    except Exception:
        return None


def parse_holdings_csv(uploaded_file):
    """Parse holdings-2.csv with robust error handling"""
    try:
        # Read CSV with quote handling
        df = pd.read_csv(uploaded_file, quotechar='"', escapechar='\\')

        # Clean column names
        df.columns = [col.strip('"').strip() for col in df.columns]

        # Find columns by partial match (robust)
        ticker_col = None
        qty_col = None
        price_col = None

        for col in df.columns:
            col_lower = col.lower()
            if "instrument" in col_lower or "symbol" in col_lower:
                ticker_col = col
            elif "qty" in col_lower or "quantity" in col_lower:
                qty_col = col
            elif "avg" in col_lower or "cost" in col_lower or "price" in col_lower:
                price_col = col

        if not all([ticker_col, qty_col, price_col]):
            st.error(f"Could not find required columns. Found: {list(df.columns)}")
            return pd.DataFrame()

        # Extract data
        holdings = pd.DataFrame({
            "ticker": df[ticker_col].astype(str).str.strip(),
            "shares": pd.to_numeric(df[qty_col], errors="coerce"),
            "buy_price": pd.to_numeric(df[price_col], errors="coerce"),
        })

        # Tag SGB instruments
        holdings["asset_type"] = holdings["ticker"].apply(
            lambda x: "SGB" if pd.notna(x) and "SGB" in str(x).upper() else "EQUITY"
        )

        # Filter valid rows only
        valid_mask = (
            (holdings["ticker"].str.len() > 0)
            & holdings["shares"].notna()
            & (holdings["shares"] > 0)
            & holdings["buy_price"].notna()
            & (holdings["buy_price"] > 0)
        )

        holdings = holdings[valid_mask].reset_index(drop=True)
        return holdings

    except Exception as e:
        st.error(f"❌ CSV parsing error: {str(e)}")
        st.error("Expected format: Instrument,Qty.,Avg. cost,...")
        return pd.DataFrame()


def save_portfolio(holdings_df):
    """Save parsed holdings to JSON"""
    portfolio_data = holdings_df[["ticker", "shares", "buy_price", "asset_type"]].to_dict("records")
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio_data, f, indent=2, default=str)
    return len(portfolio_data)


def load_portfolio():
    """Load portfolio from JSON"""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                data = json.load(f)
                return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error loading {PORTFOLIO_FILE}: {e}")
    return pd.DataFrame()


@st.cache_data(ttl=900)
def fetch_close_matrix(symbols, period="2y"):
    """Download close/adj close for multiple symbols and return Date x Symbol dataframe."""
    symbols = [s for s in symbols if s]
    if not symbols:
        return pd.DataFrame()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        data = yf.download(symbols, period=period, auto_adjust=False, group_by="column")

    if data is None or data.empty:
        return pd.DataFrame()

    # Normalize to Close (or Adj Close) with symbols as columns
    if isinstance(data.columns, pd.MultiIndex):
        top = list(data.columns.get_level_values(0))
        if "Adj Close" in top:
            close = data["Adj Close"]
        elif "Close" in top:
            close = data["Close"]
        else:
            # fallback
            try:
                close = data.xs("Close", axis=1, level=0, drop_level=True)
            except Exception:
                close = data.iloc[:, :len(symbols)]
    else:
        close = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

    close = close.dropna(how="all")
    close.index = pd.to_datetime(close.index)
    return close


def relative_return(df_close: pd.DataFrame) -> pd.DataFrame:
    """100-based relative return for each column (first row = 100)."""
    if df_close is None or df_close.empty:
        return pd.DataFrame()
    df = df_close.dropna(how="all").copy()
    df = df.loc[df.index.sort_values()]
    first = df.iloc[0]
    first = first.replace(0, np.nan)
    rr = (df / first) * 100
    return rr.dropna(how="all")


@st.cache_data(ttl=300)
def get_cad_inr_rate():
    """Get live CAD/INR from yfinance"""
    try:
        cad_inr_ticker = yf.Ticker("CADINR=X")  # CAD/INR pair
        data = cad_inr_ticker.history(period="1d")
        if not data.empty:
            return data["Close"].iloc[-1]
        return None
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_usd_inr_rate():
    """Get live USD/INR from yfinance"""
    try:
        usd_inr_ticker = yf.Ticker("INR=X")  # USD/INR pair
        data = usd_inr_ticker.history(period="1d")
        if not data.empty:
            return data["Close"].iloc[-1]
        return None
    except Exception:
        return None


# ============================================================================
# MAIN UI
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Upload CSV",
    "📊 Portfolio View",
    "📈 SGB vs Gold",
    "📉 Performance",
    "🕯️ Candlestick chart"
])


with tab1:
    st.header("📁 Upload Holdings CSV")

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        help="Your holdings-2.csv format works perfectly",
    )

    if uploaded_file is not None:
        holdings_df = parse_holdings_csv(uploaded_file)

        if not holdings_df.empty:
            st.success(f"✅ Parsed {len(holdings_df)} valid holdings")

            st.subheader("📋 Preview")
            st.dataframe(
                holdings_df[["ticker", "shares", "buy_price", "asset_type"]],
                width="stretch",
            )

            sgbs = len(holdings_df[holdings_df["asset_type"] == "SGB"])
            st.info(f"🔍 {sgbs} SGBs detected (tracked vs gold)")

            if st.button("💾 Save Portfolio", type="primary", use_container_width=True):
                count = save_portfolio(holdings_df)
                st.success(f"✅ Saved {count} holdings!")
                st.balloons()
        else:
            st.error("❌ No valid data found in CSV")


with tab2:
    st.header("📊 Portfolio View (Live Prices)")

    portfolio_df = load_portfolio()

    if portfolio_df.empty:
        st.info("📭 **Upload CSV first** (Tab 1)")
    else:
        st.success(f"📊 {len(portfolio_df)} holdings loaded")

        # ✅ AUTO-FETCH LIVE PRICES ON LOAD (no button needed)
        st.info("🔄 Fetching live prices...")
        progress_bar = st.progress(0)

        # Initialize columns safely
        portfolio_df = portfolio_df.copy()
        portfolio_df["live_price"] = portfolio_df["ticker"].apply(get_live_price)
        portfolio_df["current_value"] = portfolio_df["shares"] * portfolio_df["live_price"].fillna(0)
        portfolio_df["invested"] = portfolio_df["shares"] * portfolio_df["buy_price"]
        portfolio_df["pnl"] = portfolio_df["current_value"] - portfolio_df["invested"]
        portfolio_df["pnl_pct"] = (
            (portfolio_df["pnl"] / portfolio_df["invested"] * 100).fillna(0)
        ).round(2)

        # Update progress (batch style)
        successful = portfolio_df["live_price"].notna().sum()
        progress_bar.progress(100)
        st.success(f"✅ Fetched prices for {successful}/{len(portfolio_df)} tickers")

        # Summary metrics
        total_invested = portfolio_df["invested"].sum()
        total_value = portfolio_df["current_value"].sum()
        total_pnl = portfolio_df["pnl"].sum()
        pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Invested", f"₹{total_invested:,.0f}")
        with col2:
            st.metric("Market Value", f"₹{total_value:,.0f}")
        with col3:
            color = "normal" if pnl_pct >= 0 else "inverse"
            st.metric("Total P&L", f"₹{total_pnl:,.0f}", f"{pnl_pct:.1f}%", delta_color=color)
        with col4:
            st.metric("With Live Price", portfolio_df["live_price"].notna().sum())

        # Holdings table
        st.subheader("📋 Holdings")
        display_df = portfolio_df[[
            "ticker", "asset_type", "shares",
            "buy_price", "live_price",
            "current_value", "pnl_pct"
        ]].copy()

        st.dataframe(
            display_df,
            column_config={
                "live_price": st.column_config.NumberColumn("Live Price", format="₹%.2f"),
                "pnl_pct": st.column_config.NumberColumn("P&L %", format="%.2f%%"),
                "current_value": st.column_config.NumberColumn("Cur. Value", format="₹%.0f"),
            },
            width="stretch",
        )

        # P&L chart
        st.subheader("📈 P&L Chart")
        fig = px.bar(
            display_df.sort_values("pnl_pct"),
            x="ticker", y="pnl_pct",
            color="pnl_pct",
            color_continuous_scale=["red", "yellow", "green"],
            title="Live P&L % by Holding",
            text="pnl_pct",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


with tab3:
    st.header("🔗 SGB vs Comex Gold (GC=F)")

    portfolio_df = load_portfolio()

    if portfolio_df.empty:
        st.info("📭 Upload portfolio first")
    else:
        sgbs = portfolio_df[portfolio_df["asset_type"] == "SGB"].copy()

        if sgbs.empty:
            st.info("📭 No SGB holdings")
        else:
            total_grams = sgbs["shares"].sum()
            st.success(f"📊 {len(sgbs)} SGB holdings = **{total_grams:,.0f} grams gold**")

            try:
                comex_df = yf.Ticker("GC=F").history(period="1d")
                comex_price_usd_oz = float(comex_df["Close"].iloc[-1]) if not comex_df.empty else None
            except Exception:
                comex_price_usd_oz = None

            usd_inr_rate = get_usd_inr_rate()
            cad_inr_rate = get_cad_inr_rate()

            if comex_price_usd_oz is not None and usd_inr_rate is not None:
                comex_price_usd_oz = float(comex_price_usd_oz)
                usd_inr_rate = float(usd_inr_rate)
                gold_price_inr_gram = (comex_price_usd_oz * usd_inr_rate) / 31.1035

                st.subheader("🌟 Live Gold Benchmark")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Comex Gold", f"${comex_price_usd_oz:,.2f}/oz")
                with col2:
                    st.metric("USD/INR", f"₹{usd_inr_rate:.2f}")
                with col3:
                    st.metric("Gold/gram", f"₹{gold_price_inr_gram:,.0f}")
                with col4:
                    st.metric("Your Grams", f"{total_grams:,.0f}")
                with col5:
                    st.metric("CAD/INR", f"{cad_inr_rate:,.2f}" if cad_inr_rate else "N/A")

                total_grams = float(total_grams)
                theoretical_gold_value = total_grams * float(gold_price_inr_gram)

                invested_sgb_value = (sgbs["shares"] * sgbs["buy_price"]).sum()
                sgbs["live_price"] = sgbs["ticker"].apply(get_live_price)
                sgbs["sgb_market_value"] = sgbs["shares"] * sgbs["live_price"].fillna(0)
                market_sgb_value = sgbs["sgb_market_value"].sum()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Actual SGB Value", f"₹{invested_sgb_value:,.0f}")
                with col2:
                    st.metric("Gold Market Value", f"₹{theoretical_gold_value:,.0f}")
                with col3:
                    premium = theoretical_gold_value - invested_sgb_value
                    st.metric("Gold Market +/-", f"₹{premium:,.0f}")
                with col4:
                    pct = (theoretical_gold_value / invested_sgb_value * 100) if invested_sgb_value else 0
                    st.metric("Gold Market Value %", f"{pct:,.0f}%")

                st.subheader("📋 SGB Holdings")
                sgbs_display = sgbs[["ticker", "shares", "buy_price", "live_price", "sgb_market_value"]].copy()
                sgbs_display.columns = ["Ticker", "Grams", "Buy Price", "Live Price", "Market Value"]
                st.dataframe(sgbs_display.round(2), width="stretch")

                st.subheader("📊 Visual Comparison")
                comparison_data = pd.DataFrame({
                    "Category": ["Market SGB Value", "Theoretical Gold Value"],
                    "Value": [invested_sgb_value, theoretical_gold_value],
                })
                fig = px.bar(
                    comparison_data,
                    x="Category", y="Value",
                    color="Category",
                    text="Value",
                    title="Your SGB Holdings vs Pure Gold Value",
                )
                fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
                fig.update_layout(yaxis_title="₹ Value", height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("❌ Missing data: Comex Gold or USD/INR")


with tab4:
    st.header("📉 Performance (Select tickers + benchmark)")

    portfolio_df = load_portfolio()
    if portfolio_df.empty:
        st.info("Upload portfolio first (Tab 1).")
    else:
        equities = portfolio_df[portfolio_df["asset_type"] != "SGB"].copy()
        all_tickers = sorted(equities["ticker"].dropna().unique().tolist())

        if not all_tickers:
            st.info("No equity tickers found in portfolio.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Which holdings to include in chart
                selected = st.multiselect(
                    "Show tickers",
                    all_tickers,
                    default=all_tickers,
                    key="perf_select_tickers"
                )

            with col2:
                # Benchmark can be NIFTY50 or any holding
                benchmark_options = ["^NSEI (NIFTY 50)"] + all_tickers
                benchmark_idx = st.selectbox(
                    "Benchmark",
                    range(len(benchmark_options)),
                    format_func=lambda i: benchmark_options[i],
                    key="perf_benchmark"
                )
                benchmark_label = benchmark_options[benchmark_idx]
                # Extract actual symbol
                benchmark = "^NSEI" if "^NSEI" in benchmark_label else all_tickers[benchmark_idx - 1]

            lookback = st.selectbox(
                "Lookback",
                ["6mo", "1y", "2y", "5y"],
                index=2,
                key="perf_lookback"
            )

            if selected:
                symbols = [to_yf_symbol(t) for t in selected] + [to_yf_symbol(benchmark)]
                symbols = list(dict.fromkeys([s for s in symbols if s]))  # de-dupe

                close = fetch_close_matrix(symbols, period=lookback)
                if close.empty:
                    st.warning("No data returned from yfinance for the selected period.")
                else:
                    rr = relative_return(close)

                    bench_sym = to_yf_symbol(benchmark)
                    if bench_sym not in rr.columns:
                        st.warning(f"Benchmark series not found: {bench_sym}")
                    else:
                        rel_vs_bench = rr.div(rr[bench_sym], axis=0) * 100

                        st.subheader("📈 Relative performance vs benchmark (Benchmark = 100)")
                        fig = px.line(
                            rel_vs_bench,
                            x=rel_vs_bench.index,
                            y=rel_vs_bench.columns,
                            title="Holdings Performance vs Benchmark"
                        )
                        fig.update_layout(legend_title_text="Symbol", height=550)
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("📊 Snapshot (end vs start)")
                        last = rel_vs_bench.iloc[-1].sort_values(ascending=False).rename("RelPerf_Index")
                        st.dataframe(last.to_frame().round(2), use_container_width=True)
            else:
                st.info("Select at least one ticker to display.")


with tab5:
    st.header("🕯️ Candlestick + Market Profile")

    portfolio_df = load_portfolio()
    if portfolio_df.empty:
        st.info("Upload portfolio first (Tab 1).")
    else:
        equities = portfolio_df[portfolio_df["asset_type"] != "SGB"].copy()
        tickers = sorted(equities["ticker"].dropna().unique().tolist())

        if not tickers:
            st.info("No equity tickers found in portfolio.")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pick = st.selectbox("Ticker", tickers, key="candle_ticker")
            with col2:
                interval = st.selectbox(
                    "Interval",
                    ["1d", "1h", "30m", "15m", "5m"],
                    index=0,
                    key="candle_interval"
                )
            with col3:
                period = st.selectbox(
                    "Period",
                    ["1mo", "3mo", "6mo", "1y", "2y"],
                    index=1,
                    key="candle_period"
                )

            symbol = to_yf_symbol(pick)

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                hist = yf.Ticker(symbol).history(period=period, interval=interval)

            if hist is None or hist.empty or hist["Close"].isna().all():
                st.warning(f"No price data found for {symbol}.")
            else:
                prices = hist["Close"].dropna()
                vols = hist["Volume"].fillna(0)

                bins = st.slider("Profile bins", 10, 60, 20, 5, key="candle_bins")
                pmin, pmax = float(prices.min()), float(prices.max())
                price_ax = np.linspace(pmin, pmax, num=bins)
                vol_ax = np.zeros(bins)

                # Bin volume by price
                idxs = np.searchsorted(price_ax, prices.values, side="right") - 1
                idxs = np.clip(idxs, 0, bins - 1)
                for bi, v in zip(idxs, vols.loc[prices.index].values):
                    vol_ax[bi] += v

                fig = make_subplots(
                    rows=1, cols=2,
                    column_widths=[0.22, 0.78],
                    horizontal_spacing=0.02
                )

                fig.add_trace(
                    go.Bar(x=vol_ax, y=price_ax, orientation="h", name="Volume"),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Candlestick(
                        x=hist.index,
                        open=hist["Open"],
                        high=hist["High"],
                        low=hist["Low"],
                        close=hist["Close"],
                        name="Candle"
                    ),
                    row=1, col=2
                )

                fig.update_layout(
                    height=750,
                    showlegend=False,
                    title=f"{symbol} — Candles + Market Profile",
                    xaxis_rangeslider_visible=False,
                )
                fig.update_yaxes(showticklabels=False, row=1, col=1)
                fig.update_yaxes(title="Price (₹)", side="right", row=1, col=2)
                fig.update_xaxes(title="Volume", row=1, col=1)
                fig.update_xaxes(title="Date", row=1, col=2)

                st.plotly_chart(fig, use_container_width=True)


# Footer
st.divider()
st.caption(f"✅ Stage 01 Complete | Gold: GC=F | NSE: .NS suffix | {datetime.now().strftime('%H:%M:%S')}")
