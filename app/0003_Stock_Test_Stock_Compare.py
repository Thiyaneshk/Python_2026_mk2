import plotly.graph_objects as go
import streamlit as st
import pandas_ta as ta
import pandas as pd
import yfinance as yf
import numpy as np
import talib
import os
import io
import requests
from datetime import datetime


def _to_1d(arr_like):
    arr = np.asarray(arr_like)
    if arr.ndim > 1:
        arr = arr.squeeze()
    return arr.astype(float)


def _safe_talib(func, *arrays, **kwargs):
    converted = []
    for a in arrays:
        ca = _to_1d(a)
        converted.append(ca)
    if not converted:
        return np.array([])
    n = converted[0].shape[0]
    for ca in converted:
        if ca.shape[0] != n:
            return np.full(n, np.nan)
    tp = kwargs.get("timeperiod", 0) or 0
    if n < max(1, int(tp)):
        return np.full(n, np.nan)
    try:
        res = func(*converted, **kwargs)
    except Exception:
        return np.full(n, np.nan)
    res = np.asarray(res)
    if res.ndim == 0:
        return np.full(n, res)
    if res.shape[0] != n:
        out = np.full(n, np.nan)
        out[-res.shape[0]:] = res
        return out
    return res

def filter_todays_from_csv_v5():
    print("1")
    todays_df = pd.read_csv('Nifty50.csv', index_col=False)
    print("2")
    todays_date = datetime.strptime(todays_df['Date'].max(),'%d-%m-%Y')
    print("3 - todays_date is " + str(todays_date))
    sday_df = todays_df[todays_df['Date'] != todays_date]
    print(sday_df)
    print("4")
    # sday_date = datetime.strptime(sday_df['Date'].max(),'%d-%m-%Y')
    sday_date = datetime.strptime(sday_df['Date'],'%d-%m-%Y')
    print ("Todays date    : ",todays_date )
    print ("sday_date date : ",sday_date )

    sday_df = todays_df[todays_df['Date'] == sday_date]
    todays_df = todays_df[todays_df['Date'] == todays_date]

    sday_df['ST_10_3'] = sday_df['ST_10_3'].astype(float).round(2)
    sday_df['ST_20_5'] = sday_df['ST_20_5'].astype(float).round(2)

    sday_df["ST_BUY"]  =  np.where( ( sday_df['ST_10_3'] > sday_df['ST_20_5'] ) , 'BUY' , 'NO_SIG' )
    sday_df["ST_SELL"] =  np.where( ( sday_df['ST_20_5'] < sday_df['ST_10_3'] ) , 'SELL' , 'NO_SIG' )
    sday_df["ST_RAVI_LONG"] = np.where( ( sday_df['ST_10_3'] > sday_df['ST_20_5'] )
                                        & (sday_df['Close'] > sday_df['EMA_200'])  , 'BUY' ,  'NO_SIG'  )
    # print(sday_df)

    todays_df['Open'] = todays_df['Open'].astype(float).round(2)
    todays_df['High'] = todays_df['High'].astype(float).round(2)
    todays_df['Low'] = todays_df['Low'].astype(float).round(2)
    todays_df['Close'] = todays_df['Close'].astype(float).round(2)
    todays_df['Adj Close'] = todays_df['Adj Close'].astype(float).round(2)

    todays_df['EMA_13'] = todays_df['EMA_13'].astype(float).round(2)
    todays_df['EMA_34'] = todays_df['EMA_34'].astype(float).round(2)
    todays_df['EMA_50'] = todays_df['EMA_50'].astype(float).round(2)
    todays_df['EMA_200'] = todays_df['EMA_200'].astype(float).round(2)

    todays_df['AVGPRICE'] = todays_df['AVGPRICE'].astype(float).round(2)
    todays_df['ATR_14'] = todays_df['ATR_14'].astype(float).round(2)

    todays_df['RSI_14'] = todays_df['RSI_14'].astype(float).round(2)
    todays_df['ST_10_3'] = todays_df['ST_10_3'].astype(float).round(2)
    todays_df['ST_20_5'] = todays_df['ST_20_5'].astype(float).round(2)



    sday_df['Open'] = sday_df['Open'].astype(float).round(2)
    sday_df['High'] = sday_df['High'].astype(float).round(2)
    sday_df['Low'] = sday_df['Low'].astype(float).round(2)
    sday_df['Close'] = sday_df['Close'].astype(float).round(2)
    sday_df['Adj Close'] = sday_df['Adj Close'].astype(float).round(2)

    sday_df['EMA_13'] = sday_df['EMA_13'].astype(float).round(2)
    sday_df['EMA_34'] = sday_df['EMA_34'].astype(float).round(2)
    sday_df['EMA_50'] = sday_df['EMA_50'].astype(float).round(2)
    sday_df['EMA_200'] = sday_df['EMA_200'].astype(float).round(2)

    sday_df['AVGPRICE'] = sday_df['AVGPRICE'].astype(float).round(2)
    sday_df['ATR_14'] = sday_df['ATR_14'].astype(float).round(2)

    sday_df['RSI_14'] = sday_df['RSI_14'].astype(float).round(2)
    sday_df['ST_10_3'] = sday_df['ST_10_3'].astype(float).round(2)
    sday_df['ST_20_5'] = sday_df['ST_20_5'].astype(float).round(2)


    todays_df["ST_BUY"]  =  np.where( ( todays_df['ST_10_3'] > todays_df['ST_20_5'] ) , 'BUY' , 'NO_SIG' )
    todays_df["ST_SELL"] =  np.where( ( todays_df['ST_20_5'] < todays_df['ST_10_3'] ) , 'SELL' , 'NO_SIG' )

    sday_df["ST_BUY"]  =  np.where( ( sday_df['ST_10_3'] > sday_df['ST_20_5'] ) , 'BUY' , 'NO_SIG' )
    sday_df["ST_SELL"] =  np.where( ( sday_df['ST_20_5'] < sday_df['ST_10_3'] ) , 'SELL' , 'NO_SIG' )

    todays_df["ST_RAVI_LONG"] = np.where((todays_df['ST_10_3'] > todays_df['ST_20_5']) &
                                            (todays_df['Close'] > todays_df['EMA_200']) &
                                            (todays_df['Close'] > todays_df['ST_10_3']) &
                                            (todays_df['Close'] > todays_df['ST_20_5'])
                                            ,'BUY','NO_SIG')
    # sday_df["ST_RAVI_LONG"]   = np.where( ( sday_df['ST_10_3'] > sday_df['ST_20_5'] ) & (sday_df['Close'] > sday_df['EMA_200'])  , 'BUY' ,  'NO_SIG'  )

    sday_df["ST_RAVI_LONG"] = np.where( ( sday_df['ST_10_3'] > sday_df['ST_20_5']) &
                                          ( sday_df['Close'] > sday_df['EMA_200']) &
                                          ( sday_df['Close'] > sday_df['ST_10_3']) &
                                          ( sday_df['Close'] > sday_df['ST_20_5'])
                                          , 'BUY', 'NO_SIG')
    todays_df["VOL_BUY"]= np.where( ( todays_df['Volume'] > 2*todays_df['Vol_sma_20'] )
                                    &  ( todays_df['Volume'] > 2*todays_df['Vol_sma_50']  )
                                    &  ( todays_df['Volume'] > 2*todays_df['Vol_sma_100'] )
                                    &  ( todays_df['Volume'] > 2*todays_df['Vol_sma_200'] ) , 'BUY' , 'NO_SIG' )

    todays_df_scanner = pd.DataFrame(todays_df, columns=[ 'Script', 'ST_RAVI_LONG'])
    sday_df_scanner   = pd.DataFrame(sday_df  , columns=[ 'Script', 'ST_RAVI_LONG'])

    df_diff = pd.concat([todays_df_scanner, sday_df_scanner]).drop_duplicates(keep=False)
    print(df_diff)

    print('Output ################' )
    scanned_df2 = df_diff['Script'].to_numpy()
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$' )
    print(np.unique(scanned_df2))
    final_scripts= np.unique(scanned_df2)
    print('Output ################1' )

    rslt_df1 = sday_df[sday_df['Script'].isin(final_scripts)]
    rslt_df2 = todays_df[todays_df['Script'].isin(final_scripts)]
    Final_result_data_frame = pd.concat([rslt_df1, rslt_df2], ignore_index=True, sort=False)
    print(Final_result_data_frame)
    scanner_df= pd.DataFrame(rslt_df1, columns=[ 'Script','Close','ST_10_3','ST_20_5','EMA_200', 'ST_BUY','ST_SELL','ST_RAVI_LONG'])
    myfile = "Super_trend_Scanner.csv"
    if os.path.isfile(myfile):
        os.remove(myfile)
    else:
        print("Error: %s file not found" % myfile)

    if not os.path.isfile(myfile):
        # Final_result_data_frame.to_csv(myfile, index=False, header='column_names')
        scanner_df.to_csv(myfile, index=False, header='column_names')
    else:
        # Final_result_data_frame.to_csv(myfile, index=False, mode='a', header=False)
        scanner_df.to_csv(myfile, index=False, mode='a', header=False)

    myfile = "Nifty500_data.csv"
    if os.path.isfile(myfile):
        os.remove(myfile)
    else:
        print("Error: %s file not found" % myfile)

    if not os.path.isfile(myfile):
        # Final_result_data_frame.to_csv(myfile, index=False, header='column_names')
        todays_df.to_csv(myfile, index=False, header='column_names')
    else:
        # Final_result_data_frame.to_csv(myfile, index=False, mode='a', header=False)
        todays_df.to_csv(myfile, index=False, mode='a', header=False)

    Volume_script_df = todays_df[todays_df['VOL_BUY'] == 'BUY']
    # print(Volume_script_df)

    myfile = "Nifty500_Volume_script.csv"
    if os.path.isfile(myfile):
        os.remove(myfile)
    else:
        print("Error: %s file not found" % myfile)

    if not os.path.isfile(myfile):
        Volume_script_df.to_csv(myfile, index=False, header='column_names')
    else:
        Volume_script_df.to_csv(myfile, index=False, mode='a', header=False)

def filter_Volume_scanner():
    Nifty500_today_df = pd.read_csv('Nifty500_today.csv')
    Volume_script_df = Nifty500_today_df[Nifty500_today_df['VOL_BUY'] == 'BUY']
    print(Volume_script_df)

    myfile = "Nifty500_Volume_script.csv"
    if os.path.isfile(myfile):
        os.remove(myfile)
    else:
        print("Error: %s file not found" % myfile)

    if not os.path.isfile(myfile):
        Volume_script_df.to_csv(myfile, index=False, header='column_names')
    else:
        Volume_script_df.to_csv(myfile, index=False, mode='a', header=False)

def streamLit_NSE_data_without_Plotly_v1(script_name,l_period,l_interval):

    precision=2
    # df = yf.download("^NSEI", period="5d", interval="5m")
    # df = yf.download(f"{script_name}.NS", period="5d", interval="5m")
    df = yf.download(f"{script_name}.NS", period = l_period, interval = l_interval)

    df["EMA_13"] = _safe_talib(talib.EMA, df["Close"], timeperiod=13)
    df["EMA_34"] = _safe_talib(talib.EMA, df["Close"], timeperiod=34)
    df["EMA_50"] = _safe_talib(talib.EMA, df["Close"], timeperiod=50)
    df["EMA_200"] = _safe_talib(talib.EMA, df["Close"], timeperiod=200)

    df["RSI_14"] = _safe_talib(talib.RSI, df["Close"], timeperiod=14)
    df["ATR_14"] = _safe_talib(talib.ATR, df["High"], df["Low"], df["Close"], timeperiod=14)
    st_short = ta.supertrend(df['High'], df['Low'], df['Close'], 10, 3)
    st_long = ta.supertrend(df['High'], df['Low'], df['Close'], 20, 5)

    st_long_df = st_long[["SUPERT_20_5.0", "SUPERTd_20_5.0", "SUPERTl_20_5.0", "SUPERTs_20_5.0"]]
    st_short_df = st_short[["SUPERT_10_3.0", "SUPERTd_10_3.0", "SUPERTl_10_3.0", "SUPERTs_10_3.0"]]

    st_long_df["ST_20_5"] = np.where(st_long["SUPERTd_20_5.0"].between(0, 30),
                                     st_long["SUPERTl_20_5.0"], st_long["SUPERTs_20_5.0"])
    st_short_df["ST_10_3"] = np.where(st_short["SUPERTd_10_3.0"].between(0, 30),
                                      st_short["SUPERTl_10_3.0"], st_short["SUPERTs_10_3.0"])

    final_df = pd.concat([df, st_short_df, st_long_df], axis=1)
    final_df = final_df.iloc[-100:]
    final_df=final_df.round({"Open": precision, "High": precision,  "Low": precision, "Close": precision ,
                             "RSI_14": precision, "ATR_14": precision, "ST_10_3": precision,"ST_20_5": precision,
                             "EMA_13": precision, "EMA_34": precision, "EMA_50": precision, "EMA_200": precision})
    final_df = final_df.drop(columns=['Adj Close','Volume', 'SUPERTd_10_3.0', 'SUPERTl_10_3.0', 'SUPERTs_10_3.0', 'SUPERT_20_5.0', 'SUPERTd_20_5.0', 'SUPERTl_20_5.0', 'SUPERTs_20_5.0','SUPERT_10_3.0'])
    final_df['Open'] = final_df['Open'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['High'] = final_df['High'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['Low'] = final_df['Low'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['Close'] = final_df['Close'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['EMA_13'] = final_df['EMA_13'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['EMA_34'] = final_df['EMA_34'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['EMA_50'] = final_df['EMA_50'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['EMA_200'] = final_df['EMA_200'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['RSI_14'] = final_df['RSI_14'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['ATR_14'] = final_df['ATR_14'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['ST_10_3'] = final_df['ST_10_3'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['ST_20_5'] = final_df['ST_20_5'].astype(str).apply(lambda x: x.replace('.0', ''))

    # final_df.insert(0, 'Script', script_name)
    # print(final_df)

    st.write(final_df.style.highlight_max(axis=0))

    # if not os.path.isfile('Nifty50.csv'):
    #     final_df.to_csv('Nifty50.csv', index=True, header='column_names')
    # else:
    #     final_df.to_csv('Nifty50.csv', index=True, mode='a', header=False)

def streamLit_NSE_data_with_Plotly_v2(script_name,l_period,l_interval):

    precision=2
    # df = yf.download("^NSEI", period="5d", interval="5m")
    # df = yf.download(f"{script_name}.NS", period="5d", interval="5m")
    df = yf.download(f"{script_name}.NS", period = l_period, interval = l_interval)

    df["EMA_13"] = _safe_talib(talib.EMA, df["Close"], timeperiod=13)
    df["EMA_34"] = _safe_talib(talib.EMA, df["Close"], timeperiod=34)
    df["EMA_50"] = _safe_talib(talib.EMA, df["Close"], timeperiod=50)
    df["EMA_200"] = _safe_talib(talib.EMA, df["Close"], timeperiod=200)

    df["RSI_14"] = _safe_talib(talib.RSI, df["Close"], timeperiod=14)
    df["ATR_14"] = _safe_talib(talib.ATR, df["High"], df["Low"], df["Close"], timeperiod=14)
    st_short = ta.supertrend(df['High'], df['Low'], df['Close'], 10, 3)
    st_long = ta.supertrend(df['High'], df['Low'], df['Close'], 20, 5)

    st_long_df = st_long[["SUPERT_20_5.0", "SUPERTd_20_5.0", "SUPERTl_20_5.0", "SUPERTs_20_5.0"]]
    st_short_df = st_short[["SUPERT_10_3.0", "SUPERTd_10_3.0", "SUPERTl_10_3.0", "SUPERTs_10_3.0"]]

    st_long_df["ST_20_5"] = np.where(st_long["SUPERTd_20_5.0"].between(0, 30),
                                     st_long["SUPERTl_20_5.0"], st_long["SUPERTs_20_5.0"])
    st_short_df["ST_10_3"] = np.where(st_short["SUPERTd_10_3.0"].between(0, 30),
                                      st_short["SUPERTl_10_3.0"], st_short["SUPERTs_10_3.0"])

    final_df = pd.concat([df, st_short_df, st_long_df], axis=1)
    final_df = final_df.iloc[-100:]
    final_df=final_df.round({"Open": precision, "High": precision,  "Low": precision, "Close": precision ,
                             "RSI_14": precision, "ATR_14": precision, "ST_10_3": precision,"ST_20_5": precision,
                             "EMA_13": precision, "EMA_34": precision, "EMA_50": precision, "EMA_200": precision})
    final_df = final_df.drop(columns=['Adj Close',
                                      # 'Volume',
                                      'SUPERTd_10_3.0', 'SUPERTl_10_3.0', 'SUPERTs_10_3.0', 'SUPERT_20_5.0', 'SUPERTd_20_5.0', 'SUPERTl_20_5.0', 'SUPERTs_20_5.0','SUPERT_10_3.0'])
    final_df['Open'] = final_df['Open'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['High'] = final_df['High'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['Low'] = final_df['Low'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['Close'] = final_df['Close'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['EMA_13'] = final_df['EMA_13'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['EMA_34'] = final_df['EMA_34'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['EMA_50'] = final_df['EMA_50'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['EMA_200'] = final_df['EMA_200'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['RSI_14'] = final_df['RSI_14'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['ATR_14'] = final_df['ATR_14'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['ST_10_3'] = final_df['ST_10_3'].astype(str).apply(lambda x: x.replace('.0', ''))
    final_df['ST_20_5'] = final_df['ST_20_5'].astype(str).apply(lambda x: x.replace('.0', ''))

    # final_df.insert(0, 'Script', script_name)
    # print(final_df)

    st.write(final_df.style.highlight_max(axis=0))
    # fig = go.Figure(data=go.Ohlc(x=final_df.index,
    #                              open=final_df['Open'],
    #                              high=final_df['High'],
    #                              low=final_df['Low'],
    #                              close=final_df['Close']))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=final_df.index,
                                 open = final_df['Open'],high = final_df['High'], low = final_df['Low'], close = final_df['Close']
                                 ))
                                 # , name="OHLC"), row=1, col=1)


    st.plotly_chart(fig)

    # if not os.path.isfile('Nifty50.csv'):
    #     final_df.to_csv('Nifty50.csv', index=True, header='column_names')
    # else:
    #     final_df.to_csv('Nifty50.csv', index=True, mode='a', header=False)

if __name__ == '__main__':
    url = 'https://archives.nseindia.com/content/indices/ind_nifty50list.csv'
    # url = 'https://www1.nseindia.com/content/indices/ind_nifty50list.csv'
    # url = 'https://www1.nseindia.com/content/indices/ind_nifty500list.csv'
    s = requests.get(url).content
    script_df = pd.read_csv(io.StringIO(s.decode('utf-8')))

    Scripts_dropdown = script_df.Symbol.unique().tolist()
    l_Script = st.sidebar.selectbox('Select Script', Scripts_dropdown)
    l_interval = st.sidebar.selectbox("Select interval:",
                                     ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"))
    l_Period = st.sidebar.selectbox("Select Period:",
                                   ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"))
    st.write( " ## " + l_Script + "  --> " + l_interval + " data for last " + l_Period)

    # streamLit_NSE_data_without_Plotly_v1(l_Script,l_Period,l_interval)
    streamLit_NSE_data_with_Plotly_v2(l_Script,l_Period,l_interval)