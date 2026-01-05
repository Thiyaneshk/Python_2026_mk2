import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="ITC.NS Stock Dashboard", layout="wide")
st.title("🔥 ITC.NS Stock Dashboard")

@st.cache_data(ttl=300)  # Cache for 5 mins
def fetch_stock_data(symbol="ITC.NS"):
    ticker = yf.Ticker(symbol)
    
    # Latest info
    info = ticker.info
    hist = ticker.history(period="1y")
    dividends = ticker.dividends.tail(5)
    financials = ticker.quarterly_financials
    news = ticker.news[:5]
    
    return info, hist, dividends, financials, news

info, hist, dividends, financials, news = fetch_stock_data()

# Sidebar for symbol input
symbol = st.sidebar.text_input("Stock Symbol", value="ITC.NS")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

col1, col2, col3, col4 = st.columns(4)
current_price = info.get('currentPrice', hist['Close'][-1])
prev_close = info.get('previousClose', hist['Close'][-2] if len(hist)>1 else 0)
change = current_price - prev_close
pct_change = (change / prev_close) * 100 if prev_close else 0

col1.metric("Current Price", f"₹{current_price:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
col2.metric("Market Cap", f"₹{info.get('marketCap', 0):,.0f}", delta=None)
col3.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
col4.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A'):.2f}")

# Key Stats Table
st.subheader("📊 Key Metrics")
stats_data = {
    'Metric': ['52W High', '52W Low', 'Volume', 'Beta', 'EPS (TTM)', 'Book Value'],
    'Value': [
        f"₹{info.get('fiftyTwoWeekHigh', 'N/A'):.2f}",
        f"₹{info.get('fiftyTwoWeekLow', 'N/A'):.2f}",
        f"{info.get('volume', 'N/A'):,.0f}",
        f"{info.get('beta', 'N/A'):.2f}",
        f"{info.get('trailingEps', 'N/A'):.2f}",
        f"₹{info.get('bookValue', 'N/A'):.2f}"
    ]
}
st.table(pd.DataFrame(stats_data))

# Price Chart
st.subheader("📈 Price Chart (1 Year)")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                   subplot_titles=('Price', 'Volume'),
                   row_width=[0.7, 0.3])
fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                            low=hist['Low'], close=hist['Close'], name="Price"), row=1, col=1)
fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="Volume"), row=2, col=1)
fig.update_layout(height=600, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, width='stretch')

# Latest Updates & Price Movements
st.subheader("📰 Latest Updates & Price Movements")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Recent Price Action:**")
    recent = hist.tail(10)
    movements = []
    for i in range(1, len(recent)):
        day_change = recent['Close'].iloc[i] - recent['Close'].iloc[i-1]
        movements.append({
            'Date': recent.index[i].strftime('%Y-%m-%d'),
            'Close': f"₹{recent['Close'].iloc[i]:.2f}",
            'Change': f"₹{day_change:.2f}"
        })
    st.dataframe(pd.DataFrame(movements), width='stretch')

with col2:
    st.markdown("**Recent Dividends:**")
    if not dividends.empty:
        div_df = pd.DataFrame({
            'Date': dividends.index.strftime('%Y-%m-%d'),
            'Amount': [f"₹{x:.2f}" for x in dividends.values]
        })
        st.dataframe(div_df, width='stretch')
    else:
        st.info("No recent dividends data available.")

# Recent Developments (News)
st.subheader("🗞️ Recent Developments")
for article in news:
    with st.expander(f"**{article['title']}** - {article.get('publisher', 'N/A')}"):
        st.write(article.get('summary', 'No summary available.'))
        if 'link' in article:
            st.markdown(f"[Read more]({article['link']})")

# Key Issues / Financials
st.subheader("💼 Key Issues & Financials")
tab1, tab2 = st.tabs(["Income Statement", "Company Overview"])

with tab1:
    if not financials.empty:
        st.dataframe(financials.T.head(), width='stretch')
    else:
        st.warning("Financial data not available.")

with tab2:
    overview_data = {
        'Category': ['Sector', 'Industry', 'Employees', 'Website', 'Address'],
        'Details': [
            info.get('sector', 'N/A'),
            info.get('industry', 'N/A'),
            f"{info.get('fullTimeEmployees', 'N/A'):,}",
            info.get('website', 'N/A'),
            info.get('address', 'N/A')
        ]
    }
    st.table(pd.DataFrame(overview_data))
