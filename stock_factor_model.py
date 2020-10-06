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

from stock_utils import backfill, calc_fret, pull_price_data, load_pricing, \
    save_sp500_data, winsorize

style.use('ggplot')

############
## inputs ##
############

## start and end dates ##
start_date = '2019-12-25'
end_date = '2020-10-01'

## covariance matrix parameters ##
half_life = 18 # half_life of covariance matrix, in weeks


##################################
## pull data from yahoo finance ##
##################################


price_data, SP_data = load_pricing(start = start_date, end = end_date, refresh_sp = True, 
                                   refresh_pricing = True, save_pricing = False)


######################################
## create sector based factor model ##
######################################

## list of dates for weekly returns, friday - friday ##

dates = pd.date_range(start = start_date, end = end_date, freq = 'W-FRI')

## calculate relative returns ##

f_ret = calc_fret(dates, price_data, SP_data)

## calculate covariance matrix ##

f_cov = f_ret.ewm(halflife = half_life).cov() ## weekly covariance
f_cov_ann = f_cov*52**0.5 ## annualized covariance matrix

## plot sector volatilities over time 
for sector in f_ret.columns:
    

        
    
    



