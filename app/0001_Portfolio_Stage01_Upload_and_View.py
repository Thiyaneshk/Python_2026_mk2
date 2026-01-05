"""
0001_Portfolio_Stage01_Upload_and_View.py - FULLY FIXED

FIXES:
1. ✅ CSV parsing: Handle quoted columns + dtype errors
2. ✅ Progress bar: Clamp 0-100, handle empty DataFrame
3. ✅ NSE suffix for ALL Indian stocks
4. ✅ Robust numeric conversion
5. ✅ Empty portfolio handling
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import yfinance as yf
import plotly.express as px

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
st.markdown("**Upload CSV → Live yfinance Prices → SGB vs Gold**")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)
# def get_live_price(ticker):
#     """Get live price with proper NSE suffix"""
#     if pd.isna(ticker) or not str(ticker).strip():
#         return None
    
#     ticker_clean = str(ticker).strip()
    
#     # NSE stocks - ALWAYS add .NS suffix (your holdings)
#     nse_keywords = ['HDFC', 'RELI', 'ITC', 'LICI', 'LUPIN', 'NIFTY', 'JUNIOR', 
#                    'GOLD', 'SILVER', 'LIQUID', 'SGB', 'GHCL', 'HAPP', 'HINDU', 
#                    'PGINVIT', 'TATVA']
    
#     if any(keyword in ticker_clean.upper() for keyword in nse_keywords):
#         ticker_symbol = f"{ticker_clean}.NS"
#     else:
#         ticker_symbol = ticker_clean
    
#     try:
#         stock = yf.Ticker(ticker_symbol)
#         data = stock.history(period="1d")
#         if not data.empty:
#             return data['Close'].iloc[-1]
#         return None
#     except:
#         return None
 
def get_live_price(ticker):
    """Smart price lookup with SGB handling"""
    if pd.isna(ticker):
        return None
    
    ticker_clean = str(ticker).strip()
    
    # ✅ SGB special handling
    sgb_map = {
        'SGBD29VIII-GB': 'GOLDINR=X',
        'SGBJAN29IX-GB': 'GOLDINR=X', 
        'SGBMAY29I': 'GOLDINR=X',
        'SGBOCT27VI-GB': 'GOLDINR=X',
        'SGBSEP31II-GB': 'GOLDINR=X'
    }
    # If this is an SGB instrument (explicit mapping or contains 'SGB'),
    # return the Comex gold price (USD/oz) converted to INR per gram.
    is_sgb = ticker_clean in sgb_map or ('SGB' in ticker_clean.upper())

    if is_sgb:
        try:
            comex = yf.Ticker("GC=F").history(period="1d")
            usd_inr = yf.Ticker("INR=X").history(period="1d")
            if not comex.empty and not usd_inr.empty:
                comex_usd_oz = comex['Close'].iloc[-1]
                usd_inr_rate = usd_inr['Close'].iloc[-1]
                # 1 troy ounce = 31.1035 grams
                price_inr_per_gram = (comex_usd_oz * usd_inr_rate) / 31.1035
                return price_inr_per_gram
            return None
        except Exception:
            return None

    # NSE stocks: always add .NS suffix for common Indian tickers
    elif any(kw in ticker_clean.upper() for kw in ['HDFC', 'RELI', 'ITC', 'LICI', 'LUPIN', 
                                                   'NIFTY', 'GOLD', 'SILVER', 'LIQUID', 
                                                   'GHCL', 'HAPP', 'HINDU', 'PGINVIT', 'TATVA','JUNIORBEES']):
        symbol = f"{ticker_clean}.NS"
    else:
        symbol = ticker_clean

    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        return data['Close'].iloc[-1] if not data.empty else None
    except Exception:
        return None


def parse_holdings_csv(uploaded_file):
    """Parse holdings-2.csv with robust error handling"""
    try:
        # Read CSV with quote handling
        df = pd.read_csv(uploaded_file, quotechar='"', escapechar='\\')
        
        # Clean column names
        df.columns = [col.strip('"') for col in df.columns]
        
        # Find columns by partial match (robust)
        ticker_col = None
        qty_col = None
        price_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'instrument' in col_lower or 'symbol' in col_lower:
                ticker_col = col
            elif 'qty' in col_lower or 'quantity' in col_lower:
                qty_col = col
            elif 'avg' in col_lower or 'cost' in col_lower or 'price' in col_lower:
                price_col = col
        
        if not all([ticker_col, qty_col, price_col]):
            st.error(f"Could not find required columns. Found: {list(df.columns)}")
            return pd.DataFrame()
        
        # Extract data
        holdings = pd.DataFrame({
            'ticker': df[ticker_col].astype(str).str.strip(),
            'shares': pd.to_numeric(df[qty_col], errors='coerce'),
            'buy_price': pd.to_numeric(df[price_col], errors='coerce')
        })
        
        # Tag SGB instruments
        holdings['asset_type'] = holdings['ticker'].apply(
            lambda x: 'SGB' if pd.notna(x) and 'SGB' in x.upper() else 'EQUITY'
        )
        
        # Filter valid rows only
        valid_mask = (
            holdings['ticker'].str.len() > 0 &
            holdings['shares'].notna() &
            (holdings['shares'] > 0) &
            holdings['buy_price'].notna() &
            (holdings['buy_price'] > 0)
        )
        
        holdings = holdings[valid_mask].reset_index(drop=True)
        
        return holdings
        
    except Exception as e:
        st.error(f"❌ CSV parsing error: {str(e)}")
        st.error("Expected format: Instrument,Qty.,Avg. cost,...")
        return pd.DataFrame()

def save_portfolio(holdings_df):
    """Save parsed holdings to JSON"""
    portfolio_data = holdings_df[['ticker', 'shares', 'buy_price', 'asset_type']].to_dict('records')
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio_data, f, indent=2, default=str)
    return len(portfolio_data)

def load_portfolio():
    """Load portfolio from JSON"""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                data = json.load(f)
                return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error loading {PORTFOLIO_FILE}: {e}")
    return pd.DataFrame()

# ============================================================================
# MAIN UI
# ============================================================================

tab1, tab2, tab3 = st.tabs(["📁 Upload CSV", "📊 Portfolio View", "📈 SGB vs Gold"])

with tab1:
    st.header("📁 Upload Holdings CSV")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Your holdings-2.csv format works perfectly"
    )
    
    if uploaded_file is not None:
        holdings_df = parse_holdings_csv(uploaded_file)
        
        if not holdings_df.empty:
            st.success(f"✅ Parsed {len(holdings_df)} valid holdings")
            
            st.subheader("📋 Preview")
            st.dataframe(
                holdings_df[['ticker', 'shares', 'buy_price', 'asset_type']],
                width='stretch'
            )
            
            sgbs = len(holdings_df[holdings_df['asset_type'] == 'SGB'])
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
        portfolio_df['live_price'] = portfolio_df['ticker'].apply(get_live_price)
        portfolio_df['current_value'] = portfolio_df['shares'] * portfolio_df['live_price'].fillna(0)
        portfolio_df['invested'] = portfolio_df['shares'] * portfolio_df['buy_price']
        portfolio_df['pnl'] = portfolio_df['current_value'] - portfolio_df['invested']
        portfolio_df['pnl_pct'] = (
            (portfolio_df['pnl'] / portfolio_df['invested'] * 100).fillna(0)
        ).round(2)
        
        # Update progress (batch style)
        successful = portfolio_df['live_price'].notna().sum()
        progress_bar.progress(100)
        st.success(f"✅ Fetched prices for {successful}/{len(portfolio_df)} tickers")
        
        # Summary metrics
        total_invested = portfolio_df['invested'].sum()
        total_value = portfolio_df['current_value'].sum()
        total_pnl = portfolio_df['pnl'].sum()
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
            st.metric("With Live Price", portfolio_df['live_price'].notna().sum())
        
        # Holdings table
        st.subheader("📋 Holdings")
        display_df = portfolio_df[['ticker', 'asset_type', 'shares', 
                                  'buy_price', 'live_price', 
                                  'current_value', 'pnl_pct']].copy()
        
        st.dataframe(
            display_df,
            column_config={
                "live_price": st.column_config.NumberColumn("Live Price", format="₹%.2f"),
                "pnl_pct": st.column_config.NumberColumn("P&L %", format="%.2f%%"),
                "current_value": st.column_config.NumberColumn("Cur. Value", format="₹%.0f")
            },
            width='stretch'
        )
        
        # P&L chart
        st.subheader("📈 P&L Chart")
        fig = px.bar(
            display_df.sort_values('pnl_pct'),
            x='ticker', y='pnl_pct',
            color='pnl_pct',
            color_continuous_scale=['red', 'yellow', 'green'],
            title="Live P&L % by Holding",
            text='pnl_pct'
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, width='stretch')



# Add this function ABOVE Tab 3 (around line 150)

@st.cache_data(ttl=300)
def get_cad_inr_rate():
    """Get live CAD/INR from yfinance"""
    try:
        cad_inr_ticker = yf.Ticker("CADINR=X")  # CAD/INR pair
        data = cad_inr_ticker.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except:
        return None
 
@st.cache_data(ttl=300)
def get_usd_inr_rate():
    """Get live USD/INR from yfinance"""
    try:
        usd_inr_ticker = yf.Ticker("INR=X")  # USD/INR pair
        data = usd_inr_ticker.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except:
        return None
        
# Then UPDATE Tab 3 section:
with tab3:
    st.header("🔗 SGB vs Comex Gold (GC=F)")
    
    portfolio_df = load_portfolio()
    
    if not portfolio_df.empty:
        sgbs = portfolio_df[portfolio_df['asset_type'] == 'SGB'].copy()
        
        if not sgbs.empty:
            total_grams = sgbs['shares'].sum()
            st.success(f"📊 {len(sgbs)} SGB holdings = **{total_grams:,.0f} grams gold**")
            
            # ✅ LIVE Comex Gold + USD/INR + CAD/INR
            # Fetch numeric values directly to avoid unit confusion
            try:
                comex_df = yf.Ticker("GC=F").history(period="1d")
                comex_price_usd_oz = float(comex_df['Close'].iloc[-1]) if not comex_df.empty else None
            except Exception:
                comex_price_usd_oz = None

            usd_inr_rate = get_usd_inr_rate()
            cad_inr_rate = get_cad_inr_rate()

            if comex_price_usd_oz is not None and usd_inr_rate is not None:
                comex_price_usd_oz = float(comex_price_usd_oz)
                usd_inr_rate = float(usd_inr_rate)
                # Gold price INR/gram (1 troy ounce = 31.1035 grams)
                gold_price_inr_gram = (comex_price_usd_oz * usd_inr_rate) / 31.1035
                
                st.subheader("🌟 Live Gold Benchmark")
                col1, col2, col3, col4 , col5 = st.columns(5)
                with col1:
                    st.metric("Comex Gold", f"${comex_price_usd_oz:,.2f}/oz")
                with col2:
                    st.metric("USD/INR", f"₹{usd_inr_rate:.2f}")
                with col3:
                    st.metric("Gold/gram", f"₹{gold_price_inr_gram:,.0f}")
                with col4:
                    st.metric("Your Grams", f"{total_grams:,.0f}")
                with col5:
                    st.metric("CAD/INR", f"{cad_inr_rate:,.2f}")
                
                # Theoretical value (ensure numeric multiplication)
                total_grams = float(total_grams)
                theoretical_gold_value = total_grams * float(gold_price_inr_gram)

                # Invested value (buy_price) and SGB market value (live)
                invested_sgb_value = (sgbs['shares'] * sgbs['buy_price']).sum()

                sgbs['live_price'] = sgbs['ticker'].apply(get_live_price)
                sgbs['sgb_market_value'] = sgbs['shares'] * sgbs['live_price'].fillna(0)
                market_sgb_value = sgbs['sgb_market_value'].sum()

                # Performance delta: compare market value vs theoretical gold value
                value_delta = market_sgb_value - theoretical_gold_value
                performance_delta = ((value_delta / theoretical_gold_value * 100) if theoretical_gold_value else 0)

                # col1, col2, col3 = st.columns(3)
                # with col1:
                #     st.metric("Invested SGB Value", f"₹{invested_sgb_value:,.0f}")
                # with col2:
                #     st.metric("Theoretical Gold", f"₹{theoretical_gold_value:,.0f}")
                # with col3:
                #     color = "normal" if performance_delta >= 0 else "inverse"
                #     st.metric(
                #         "SGB Premium/Discount", 
                #         f"{performance_delta:+.1f}%",
                #         delta=f"₹{value_delta:+,.0f}",
                #         delta_color=color
                #     )
                col1, col2, col3 , col4= st.columns(4)
                with col1:
                    st.metric("Actual SGB Value", f"₹{invested_sgb_value:,.0f}")
                with col2:
                    st.metric("Gold Market Value", f"₹{theoretical_gold_value:,.0f}")
                with col3:
                    premium = theoretical_gold_value - invested_sgb_value 
                    # premium_pct = (premium / theoretical_gold_value * 100) if theoretical_gold_value > 0 else 0
                    # color = "normal" if premium >= 0 else "inverse"
                    st.metric("Gold Market +/-", f"₹{premium:,.0f}")
                with col4:
                    st.metric("Gold Market Value", f"{theoretical_gold_value/invested_sgb_value*100:,.0f}%")

                
                # Holdings table
                st.subheader("📋 SGB Holdings")
                sgbs_display = sgbs[['ticker', 'shares', 'buy_price', 
                                   'live_price', 'sgb_market_value']].copy()
                sgbs_display.columns = ['Ticker', 'Grams', 'Buy Price', 
                                      'Live Price', 'Market Value']
                
                st.dataframe(sgbs_display.round(2), width='stretch')
                
                # Comparison chart
                # Visual comparison (FIXED Plotly syntax)
                st.subheader("📊 Visual Comparison")
                comparison_data = pd.DataFrame({
                    'Category': ['Market SGB Value', 'Theoretical Gold Value'],
                    'Value': [invested_sgb_value, theoretical_gold_value]
                })

                fig = px.bar(
                    comparison_data,
                    x='Category', 
                    y='Value',
                    color='Category',
                    color_discrete_map={
                        'Actual SGB Value': '#1f77b4', 
                        'Theoretical Gold Value': '#ff7f0e'
                    },
                    text='Value',
                    title="Your SGB Holdings vs Pure Gold Value"
                )

                # ✅ FIXED: Proper Plotly layout updates
                fig.update_traces(
                    texttemplate='₹%{text:,.0f}', 
                    textposition='outside'
                )
                fig.update_layout(
                    yaxis_title="₹ Value",
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                
            else:
                st.warning("❌ Missing data: Comex Gold or USD/INR")
                if not comex_price_usd_oz:
                    st.info("💡 GC=F not available")
                if not usd_inr_rate:
                    st.info("💡 INR=X not available")
        else:
            st.info("📭 No SGB holdings")
    else:
        st.info("📭 Upload portfolio first")


# Footer
st.divider()
st.caption(f"✅ Stage 01 Fixed | Gold: GC=F | NSE: .NS suffix | {datetime.now().strftime('%H:%M:%S')}")
