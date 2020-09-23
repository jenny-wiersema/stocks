#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:19:54 2020

@author: jenny-wiersema

Download Stock Information using Yahoo Finance
"""

###############
## libraries ##
###############

import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import style

import pandas as pd
import numpy as np
import xarray as xr

from sklearn.linear_model import LinearRegression


import yfinance as yf
import mplfinance as mpf

import bs4 as bs ## beautiful soup, html data parsing
import requests

from stock_utils import save_sp500_data, load_pricing, backfill, winsorize

style.use('ggplot')

############
## inputs ##
############

## start and end dates ##
start_date = '2020-01-01'
end_date = '2020-09-01'



##################################
## pull data from yahoo finance ##
##################################


price_data = load_pricing(start = start_date, end = end_date, refresh_sp = False, 
                          refresh_pricing = True, save_pricing = False)


######################################
## create sector based factor model ##
######################################

## list of dates, weekly returns, friday - friday ##

dates = pd.date_range(start = start_date, end = end_date, freq = 'W-FRI')

## calculate log returns ##

for date_0, date_1 in zip(dates[:-1], dates[1:]):
    p0 = price_data.sel(date = date_0, drop = True)['Adj Close']
    p1 = price_data.sel(date = date_1, drop = True)['Adj Close']
    
    ## backfill missing data, up to 4 days
    p0 = backfill(p0, date_0, price_data)
    p1 = backfill(p1, date_1, price_data)
    
    ## calculate relative returns
    ret = p1/p0 - 1
    
    ## apply winsorization to returns
    # clip all returns +/- 5IQR from the median
    ret = winsorize(ret)

    ## calculate market factor return - using all non returns ##
    y = ret.to_series()
    X = pd.Series()
    I = ~np.isnan(y) ## remove nan data
    
    lin_reg = LinearRegression()
    lin_reg.fit(X[I],y[I])
    







all_sectors = list(set(SP_data.sector.values))


tckrs_i = SP_data.sector.ticker[SP_data.sector.values == all_sectors[i]].values



SP_data.sector.ticker[SP_data.sector.values == 'Financials']





