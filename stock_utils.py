#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 17:21:08 2020

@author: jenny-wiersema
"""

import datetime as dt
import bs4 as bs
import requests
import xarray as xr 


### Transforms dates from a yyyy-mm-dd string to a datetime object ###
def adjust_date(date):
    date_update = dt.datetime(int(date[:4]), int(date[5:7]), int(date[8:]))
    return date_update

### pulls S&P 500 data from wikipedia site
def save_sp500_data():
    # pull data from website
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'}
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                        headers=headers)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    # parse data and create lists
    tickers = []
    companies = []
    sectors = []
    subsectors = []
    
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        company_name = row.findAll('td')[1].text
        sector = row.findAll('td')[3].text
        subsector = row.findAll('td')[4].text
        
        tickers.append(ticker[:-1]) # remove /n from end of tickers
        companies.append(company_name)
        sectors.append(sector)
        subsectors.append(subsector)
        
    # compile data into dataset 
    SP_data = xr.Dataset(
    {
        'sector': ('ticker', sectors),
        'subsector': ('ticker', subsectors),
        'company': ('ticker', companies),
    },
    {'ticker': tickers},
    )
        
    return SP_data

## loads pricing data from yahoo finance
def load_pricing(start, end, refresh_sp = False, refresh_pricing = False, save_pricing = False):
    ## inputs ##
    # start: start date of factor returns to load
    # end: end date of price data to load
    # refresh_sp: if True, repull s&p data from wikipedia table
    # refresh_pricing: if True, repull pricing data from yahoo finance
    # save_pricing: if True, save pricing information into netcdf file
    ## outputs ##
    # price_xr: xarray containing pricing information    
    
    ## identify tickers in S&P500 ##
    
    if refresh_sp == True:
        SP_data = save_sp500_data()
        SP_data.to_netcdf('SP_data.nc')        
        
    elif refresh_sp == False:
        SP_data = xr.open_dataset('SP_data.nc')
        
    tckrs = list(SP_data['ticker'].values)

    ## pull pricing data ##

    if refresh_pricing == True:
        price_data = yf.download(tckrs, start=start, end=end, group_by='tickers')
        # price_data = yf.download(tckrs, period = 'max', group_by='tickers')
        price_data.index.name = 'date'
        
        # convert dataframe to xarray
        price_xr = xr.Dataset()
        idx = pd.IndexSlice
    
        for metric in list(set(price_data.columns.get_level_values(1))):
            dat = price_data.loc[:,idx[:,metric]]
            dat = dat.droplevel(axis = 1, level = 1)
            price_xr[metric] = (['date','ticker'], dat)
        
        price_xr.coords['date'] = price_data.index
        price_xr.coords['ticker'] = dat.columns
            
    elif refresh_pricing == False: 
        # load data from file
        price_xr = xr.open_dataset('price_xr.nc')
        price_xr = price_xr.sel(date = slice(start, end))
        
    if save_pricing == True:
        price_xr.to_netcdf('price_xr.nc')
     
    return price_xr


## backfill pricing information for null data
def backfill(price, date, price_data):
    for p in price:
        if np.isnan(p):
            i = 0
            while np.isnan(p) and i <= 4:
                i = i + 1
                d = date - pd.Timedelta(i, unit='d')
                if d in list(price_data['date']):
                    p_back = price_data.sel(date = d, ticker = p.ticker, drop = True)['Adj Close']
                    price.loc[dict(ticker = p.ticker)] = p_back
                    
    return price

def winsorize(ret):
    IQR = ret.quantile(0.75) - ret.quantile(0.25) #interquartile range
    ret_median = ret.quantile(0.5)
    
    cap = ret_median + 5*IQR
    floor = ret_median - 5*IQR
    
    cap = float(cap.to_pandas())
    floor = float(floor.to_pandas())

    ret = ret.clip(floor, cap)

    return ret






