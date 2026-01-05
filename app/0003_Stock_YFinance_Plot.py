import streamlit as st
import yfinance as finance
import io
import contextlib


def get_ticker(name):
    company = finance.Ticker(name)  # google
    return company


# Project Details
st.title("Build and Deploy Stock Market App Using Streamlit")
st.header("A Basic Data Science Web Application")
st.sidebar.header("Geeksforgeeks \n TrueGeeks")

company1 = get_ticker("GOOGL")
company2 = get_ticker("MSFT")

# fetches the data: Open, Close, High, Low and Volume
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    google = finance.download("GOOGL", start="2025-12-01", end="2025-12-31")
    microsoft = finance.download("MSFT", start="2025-12-01", end="2025-12-31")

# Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
# Silence yfinance console messages for missing symbols
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    data1 = company1.history(period="3mo")
if data1 is None or data1.empty:
    st.warning(f"No data found for {company1.ticker}.")
else:
    data1 = data1.drop(columns=['High', 'Low', 'Volume','Open'])


data2 = company2.history(period="3mo")
if data2 is None or data2.empty:
    st.warning(f"No data found for {company2.ticker}.")
else:
    data2 = data2

# markdown syntax
st.write("""
### Google
""")

# detailed summary on Google
st.write(company1.info['longBusinessSummary'])
st.write(google)

# plots the graph
if not (data1 is None or data1.empty):
    st.line_chart(data1.values)
else:
    st.info("No historical data to plot for Google.")

st.write("""
### Microsoft
""")
st.write(company2.info['longBusinessSummary'], "\n", microsoft)
if not (data2 is None or data2.empty):
    st.line_chart(data2.values)
else:
    st.info("No historical data to plot for Microsoft.")