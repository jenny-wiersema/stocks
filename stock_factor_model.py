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

import yfinance as yf
import mplfinance as mpf

import bs4 as bs ## beautiful soup, html data parsing
import requests

from stock_utils import backfill, calc_fret, calc_portfolio, calc_risk, \
    load_pricing, plot_bias_stats, plot_risk_envelope, plot_vol, pull_price_data, \
    save_sp500_data, winsorize

style.use('ggplot')

############
## inputs ##
############

## start and end dates ##
start_date = '2000-01-01'
end_date = '2020-12-08'

## covariance matrix parameters ##
half_life = 18 # half_life of covariance matrix, in weeks

## portfolio construction parameters ##
weights = 'equal' # weight can either be equal, or an array of weights


##################################
## pull data from yahoo finance ##
##################################


price_data, SP_data = load_pricing(start = start_date, end = end_date, refresh_sp = False, 
                                   refresh_pricing = True, save_pricing = True)


######################################
## create sector based factor model ##
######################################

## list of dates for weekly returns, friday - friday ##

dates = pd.date_range(start = start_date, end = end_date, freq = 'W-FRI')

## calculate relative returns ##

f_ret = calc_fret(dates, price_data, SP_data)

# f_ret.to_csv('f_ret.csv') 

## calculate covariance matrix ##

f_cov = f_ret.ewm(halflife = half_life).cov() ## weekly covariance
f_cov_ann = f_cov*52**0.5 ## annualized covariance matrix

## calculate sector factor volatilities

idx = pd.IndexSlice

f_vol = pd.DataFrame(index = dates[2:], columns = f_ret.columns)

for sector in f_ret.columns:
    f_vol_sector = (f_cov.loc[idx[:, sector], idx[sector]])**0.5
    f_vol_sector.index = f_vol_sector.index.droplevel(level = 1)
    f_vol[sector] = f_vol_sector


plot_vol(f_vol)

##########################################
## Calculate Portfolio Risk/Backtesting ##
##########################################

## calculate portfolio returns ##

port_ret, port_X = calc_portfolio(dates, price_data, SP_data)
# port_ret.to_csv('port_ret.csv')
# port_X.to_csv('port_X.csv')

## calculate portfolio risk ##
port_risk = calc_risk(dates[52:], port_X, port_ret, f_cov)

## risk envelope plot ##

plot_risk_envelope(port_ret['Overall'], port_risk)

        
plot_bias_stats(port_ret['Overall'], port_risk['Portfolio Risk'], 
                title = '52w Rolling Bias Stats', win = 52)
      
plot_bias_stats(port_ret['Overall'], port_risk['Portfolio Risk'], 
                title = '3Y Rolling Bias Stats', win = 156)  

    
#########################################
## Calculate Individual Risk Breakdown ##   
#########################################    
    
    
XSR = calc_risk_decomp(port_X, f_cov, date)
    
    


