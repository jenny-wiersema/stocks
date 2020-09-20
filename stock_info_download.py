#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:19:54 2020

@author: jenny-wiersema

Download Stock Information using Yahoo Finance
"""

import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials

tck = 'TSLA'

ticker = yf.Ticker(tck)
    

df = ticker.history(period="max")

tsla_df['Close'].plot(title=tck + "'s stock price")

'''__________________________________________________'''


import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib.dates as mdates

style.use('ggplot')

start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()

df = ticker.history(period="max")


tckrs = ['AMZN', 'AAPL', 'GOOG']

data = yf.download(tckrs, start="2017-01-01", end="2017-04-30", group_by='tickers')
# data = yf.download(tckrs, period = 'max', group_by='tickers')

close_values = data.loc[:,idx[:,'Adj Close']] ## pull 

data.columns.levels[1] ## identify all the columns in level 1
 
 
ma_100 = data.loc[:,idx[:,'Adj Close']].rolling(window=10).mean() #10 day moving average


ax1 = plt.subplot2grid((7,1), (0,0), rowspan=4, colspan=1)
ax2 = plt.subplot2grid((7,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.plot(data.loc[:,idx['AMZN','Adj Close']])
ax1.plot(ma_100['AMZN'])
ax2.bar(data.index, data.loc[:,idx['AMZN','Volume']])
ax1.set_title('Adjusted Close')
ax2.set_title('Volume')
plt.subplots_adjust(hspace = 0.2)
plt.setp(ax1.get_xticklabels(), ha='center', rotation=30, fontsize = 8)
plt.setp(ax2.get_xticklabels(), ha='center', rotation=30, fontsize = 8)

# ax2.set_xticklabels(labels = ax2.get_xticklabels(), rotation = 45, )
plt.show()


### open/high/low/close
df = data['AMZN']
df_ohlc = df['Adj Close'].resample('10D').ohlc()
 
 
mpf.plot(df_ohlc, type='candle', style='charles')
            title='S&P 500, Nov 2019',
            ylabel='Price ($)',
            ylabel_lower='Shares \nTraded',
            volume=True, 
            mav=(3,6,9), 
            savefig='test-mplfiance.png')