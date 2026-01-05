import requests
from io import StringIO
import pandas as pd
import streamlit as st
from datetime import date
import datetime
import calendar
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from requests.exceptions import SSLError, RequestException


## Emoji - https://emojifinder.com/

# @st.cache
def get_exp_date():
    today = datetime.date.today()
    expiry_dt = today + datetime.timedelta((3 - today.weekday()) % 7)
    input_dte = st.date_input("Enter Expiry date :", expiry_dt)
    return str.upper(input_dte.strftime('%d%b%Y'))

@st.cache_data
def Expiry_dates():
    list_exp_dates = list()
    expiry_day = "Thursday"
    start_date = datetime.date.today().strftime("%d/%m/%Y")
    end_date = datetime.date(datetime.date.today().year, 12, 31).strftime("%d/%m/%Y")

    # start_date = datetime.datetime.strptime(start, '%d/%m/%Y')
    # end_date = datetime.datetime.strptime(end, '%d/%m/%Y')

    for i in range((end_date - start_date).days):
        if calendar.day_name[(start_date + datetime.timedelta(days=i + 1)).weekday()] == expiry_day:
            # print((start_date + datetime.timedelta(days=i + 1)).strftime("%d%b%Y"))
            list_exp_dates.append((start_date + datetime.timedelta(days=i + 1)).strftime("%d%b%Y"))
    print("Printing list :", list_exp_dates)
    return list_exp_dates

@st.cache_data
def index_list():
    return ['NIFTY 50', 'Bank-Nifty', 'Nifty-500','Nifty-IT','NASDAQ']

@st.cache_data(ttl=3600)
def get_stock_list(index_option):
    """Get NSE stock list with multiple fallback URLs and caching.

    Uses the session created by `_requests_session_with_retries` for retries and headers.
    """
    urls = {
        "NIFTY 50": [
            "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
            # "https://www.nseindia.com/api/equity-master?index=equities%20market&csv=true",
        ],
        "NIFTY BANK": [
            "https://archives.nseindia.com/content/indices/ind_niftybanklist.csv"
        ],
        "NIFTY IT": [
            "https://archives.nseindia.com/content/indices/ind_niftyitlist.csv"
        ],
        "NIFTY NEXT 50": [
            "https://archives.nseindia.com/content/indices/ind_nifty_next50list.csv"
        ]
    }

    if index_option not in urls:
        return []

    session = _requests_session_with_retries()

    for url in urls[index_option]:
        try:
            resp = session.get(url, timeout=10)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.content.decode('utf-8')))

            # Try common symbol column names
            symbols = None
            for col in ("SYMBOL", "Symbol", "symbol", "Code"):
                if col in df.columns:
                    symbols = df[col].dropna().astype(str).str.strip().tolist()
                    break

            if not symbols:
                st.warning(f"No symbol column found in CSV from {url}")
                continue

            # For Indian indices (non-NASDAQ), append .NS if missing
            if 'NASDAQ' not in str(index_option).upper():
                symbols = [s + '.NS' if (not s.startswith('^') and not s.endswith('.NS')) else s for s in symbols]

            st.success(f"✅ Loaded {len(symbols)} stocks from {index_option}")
            return symbols

        except Exception as e:
            st.warning(f"❌ {url} failed: {str(e)[:200]}")
            continue

    st.error(f"❌ All URLs failed for {index_option}")
    return []

@st.cache_data
def get_nifty50_list():
    url = 'https://www1.nseindia.com/content/indices/ind_nifty50list.csv'
    # url = 'https://www1.nseindia.com/content/indices/ind_nifty500list.csv'
    script_df = _fetch_nse_csv(url)
    index_list={'Symbol':['^NSEI','^NSEBANK']}
    index_df=pd.DataFrame(index_list)
    # return index_df
    script_df['Symbol'] = script_df['Symbol']+'.NS'
    script_df = pd.concat([index_df, script_df[:]]).reset_index(drop=True)
    Scripts_dropdown = script_df.Symbol.unique().tolist()
    return Scripts_dropdown

@st.cache_data
def get_nifty500_list():
    # url = 'https://www1.nseindia.com/content/indices/ind_nifty50list.csv'
    url = 'https://www1.nseindia.com/content/indices/ind_nifty500list.csv'
    script_df = _fetch_nse_csv(url)
    index_list={'Symbol':['^NSEI','^NSEBANK']}
    index_df=pd.DataFrame(index_list)
    script_df['Symbol'] = script_df['Symbol']+'.NS'
    script_df = pd.concat([index_df, script_df[:]]).reset_index(drop=True)
    Scripts_dropdown = script_df.Symbol.unique().tolist()
    return Scripts_dropdown


@st.cache_resource
def _requests_session_with_retries(retries=3, backoff_factor=0.3, status_forcelist=(500,502,503,504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Referer": "https://www.nseindia.com/",
    })
    return session


def _fetch_nse_csv(url):
    """Fetch CSV from NSE with retries, header, and TLS fallbacks.

    Tries the original URL, then common alternate hosts, and finally falls back to
    `verify=False` if an SSL error persists (with a warning).
    Returns a pandas.DataFrame on success or raises the last exception.
    """
    session = _requests_session_with_retries()
    candidate_hosts = [
        url,
        url.replace('www1.nseindia.com', 'www.nseindia.com'),
        url.replace('www1.nseindia.com', 'archives.nseindia.com'),
        url.replace('www.nseindia.com', 'archives.nseindia.com'),
    ]
    seen = []
    last_exc = None
    for u in candidate_hosts:
        if u in seen:
            continue
        seen.append(u)
        try:
            resp = session.get(u, timeout=10)
            resp.raise_for_status()
            return pd.read_csv(StringIO(resp.content.decode('utf-8')))
        except SSLError as e:
            last_exc = e
            continue
        except RequestException as e:
            last_exc = e
            continue

    # Final fallback: try original URL with verify=False (not recommended)
    try:
        resp = session.get(url, timeout=10, verify=False)
        resp.raise_for_status()
        st.warning('Warning: fetched NSE data with certificate verification disabled.')
        return pd.read_csv(StringIO(resp.content.decode('utf-8')))
    except Exception as e:
        if last_exc:
            raise last_exc
        raise
    index_list={'Symbol':['^NSEI','^NSEBANK']}
    index_df=pd.DataFrame(index_list)
    # return index_df
    script_df['Symbol'] = script_df['Symbol']+'.NS'
    script_df = pd.concat([index_df, script_df[:]]).reset_index(drop=True)
    Scripts_dropdown = script_df.Symbol.unique().tolist()
    return Scripts_dropdown

def relative_return(df):
    rel=df.pct_change()
    cumret=(1+rel).cumprod() - 1
    cumret=cumret.fillna(0)
    return cumret

@st.cache_data
def cur_year():
    return date.today().year

@st.cache_data
def cur_year_YYYY_MON_DD():
    return str(date.today().year) + '-12-31'