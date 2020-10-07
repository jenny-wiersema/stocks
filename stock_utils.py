#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 17:21:08 2020

@author: jenny-wiersema
"""

import bs4 as bs
import requests
import yfinance as yf

import pandas as pd
import numpy as np
import xarray as xr 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt



def backfill(price, date, price_data):
    """

    Backfill nan pricing data, up to 4 days prior. 

    Inputs
    ----------
    price: xarray 
        xarray containing Adj Close price data for a single date
    date: datetime
        date for pricing data
    price_data: xarray
        xarray containing all pricing data 

    Outputs
    -------
    price: xarray
        xarray containing original price input, with nan data updated to one of the previous 4 days, if available. 

    """
    
    i = 0 
    while i <=4:
        i = i + 1
        d = date - pd.Timedelta(i, unit='d')
        if d in list(price_data['date']):
            p_back = price_data.sel(date = d, drop = True)['Adj Close']
            price = price.fillna(p_back)

    return price


def calc_fret(dates, price_data, SP_data):
    """
    Calculate weekly factor returns for sector based factors
    
    ## future improvement, handle multiple types of exposures ##
    
    Inputs
    ----------
    dates: datetime
        list of weekly dates
    price_data: xarray
        xarray containing all pricing data 
    SP_data: xarray
        xarray containing SP_500 data, scraped from wikipedia page

    Outputs
    -------
    f_ret: DataFrame
        dataframe containing weekly factor returns for all sector factors

    """
    all_sectors = list(set(SP_data['sector'].values))
    all_sectors.sort()
    f_ret = pd.DataFrame(index = dates[1:], columns = all_sectors)

    for date_0, date_1 in zip(dates[:-1], dates[1:]):
        
        if date_0.year != date_1.year:
            print('Starting ' + str(date_1.year))
        
        ## load pricing data. If entire day is missing, use previous day (Holiday Handling)
        p0 = pull_price_data(date_0, price_data)
        p1 = pull_price_data(date_1, price_data)
        
        ## backfill missing data, up to 4 days.
        p0 = backfill(p0, date_0, price_data)
        p1 = backfill(p1, date_1, price_data)
            
        ## calculate relative returns
        ret = p1/p0 - 1
        
        ## apply winsorization to returns
        # clip all returns +/- 5IQR from the median
        ret = winsorize(ret)
        # remove nan data
        ret = ret.dropna(dim = 'ticker') ## remove nan data
    
        ## define sectors ##
        sectors = SP_data.sel(ticker = ret.ticker)[['sector']].to_dataframe()
        
        onehot = OneHotEncoder(sparse = False)
        X = onehot.fit_transform(sectors)
        X = pd.DataFrame(X, index = sectors.index, columns = onehot.categories_)
    
        ## calculate sector factor returns ##
        y = np.array(ret['Adj Close'].values).reshape(-1, 1)
        
        lin_reg = LinearRegression(fit_intercept = False)
        lin_reg.fit(X,y)
        
        f_ret.loc[date_1] = lin_reg.coef_
        
    return f_ret
 
    
def calc_portfolio(dates, price_data, SP_data):
    """
    Calculate portfolio weekly returns, equally weighted

    
    Inputs
    ----------
    dates: datetime
        list of weekly dates
    price_data: xarray
        xarray containing all pricing data 
    SP_data: xarray
        xarray containing SP_500 data, scraped from wikipedia page

    Outputs
    -------
    port_ret: DataFrame
        dataframe containing weekly portfolio returns, both overall and sector subportfolios
    port_X: DataFrame
        dataframe containing weekly portfolio exposures by sector
    """
    
    all_sectors = list(set(SP_data['sector'].values))
    all_sectors.sort()
    port_ret = pd.DataFrame(index = dates[1:], columns = all_sectors)
    port_ret_overall = pd.DataFrame(index = dates[1:], columns = ['Overall'])

    
    port_X = pd.DataFrame(index = dates[1:], columns = all_sectors)                            

    for date_0, date_1 in zip(dates[:-1], dates[1:]):
        
        if date_0.year != date_1.year:
            print('Starting ' + str(date_1.year))
        
        ## load pricing data. If entire day is missing, use previous day (Holiday Handling)
        p0 = pull_price_data(date_0, price_data)
        p1 = pull_price_data(date_1, price_data)
        
        ## backfill missing data, up to 4 days.
        p0 = backfill(p0, date_0, price_data)
        p1 = backfill(p1, date_1, price_data)
            
        ## calculate relative returns
        ret = p1/p0 - 1
        
        ## calculate portfolio returns
        
        ret['sector'] = SP_data['sector']
        ret = ret.set_coords('sector')
        
        port_ret_overall.loc[date_1] = ret['Adj Close'].mean()
        port_ret.loc[date_1] = ret['Adj Close'].groupby('sector').mean()
        
        ## calculate portfolio exposures
        
        counts = ret.groupby('sector').count()['Adj Close']
        
        port_X.loc[date_1] = counts/counts.sum()
        
        

    port_ret['Overall'] = port_ret_overall
        
    return port_ret, port_X
   
    
def calc_risk(dates, port_X, port_ret, f_cov):
    """
    calculate portfolio level risk, using (XFX)^1/2

    Inputs
    ----------
    dates: datetime
        a series of dates for calculation
    port_X: DataFrame
        dataframe containing weekly portfolio exposures by sector
    port_ret: DataFrame
        dataframe containing weekly portfolio returns, both overall and sector subportfolios
    f_cov: DataFrame
        dataframe containing weekly covariance matrix

    Outputs
    -------
    port_risk: dataframe
        dataframe containing portfolio risk

    """
    port_risk = pd.DataFrame(index = dates, columns = ['Portfolio Risk'])    

    for d in dates:    
        
        X_d = port_X.loc[d]
        S_d = f_cov.loc[d]
        port_risk.loc[d] = (X_d.dot(S_d).dot(X_d))**0.5
        
    return port_risk
           

def load_pricing(start, end, refresh_sp = False, refresh_pricing = False, save_pricing = False):
    """
    load pricing information and S&P 500 information. Pricing information is pulled using yahoo 
    finance. S&P500 information is pulled from a wikipedia table. 

    Inputs
    ----------
    start: datetime
        start date for pricing information to load
    end: datetime
        end date for pricing information to load
    refresh_sp: bool, default False
        if True, pull S&P data from wikipedia table
    refresh_pricing: bool, default False
        if True, pull pricing data from yahoo finance
    save_pricing: bool, default False
        if True, save pricing information in netcdf file

    Outputs
    -------
    price_xr: xarray
        xarray containing pricing data for dates between start and end date
    SP_data: xarray
        xarray containing sector information pulled from wikipedia page

    """
    
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
     
    return price_xr, SP_data


def plot_bias_stats(port_ret, port_risk, title, win = 52):
    """
    Calculate and plot bias statistics for rolling windows
    
    Inputs
    ----------
    port_ret: DataFrame
        dataframe containing weekly portfolio returns, both overall and sector subportfolios
    port_risk: dataframe
        dataframe containing portfolio risk
    title: str
        title for plot
    win: int, default = 52
        rolling window used in bias stat calculation
    
    Outputs
    -------
    None
    
    """
    z = (port_ret/port_risk.shift(periods = 1)).clip(-3,3)
    
    bias_stats = z.rolling(window = win).std()
    
    plt.figure(figsize=(10,6))
    plt.plot(bias_stats, label = 'bias stat')
    x_min, x_max = plt.xlim()
    plt.hlines(1 + np.sqrt(2/win), xmin = x_min, xmax = x_max, color = 'k', 
               lw = 2, ls = '--', label = '95% confidence interval')
    plt.hlines(1 - np.sqrt(2/win), xmin = x_min, xmax = x_max, color = 'k', 
               lw = 2, ls = '--', label = '')
    plt.title(title)
    plt.legend(loc = 'best')
    plt.show()
    
    
def plot_risk_envelope(port_ret, port_risk):
    """
    Calculate 2sigma risk envelope and outputs plot. Risk plots use t-1 values
    as risk is a forecast and cannot include t return information. 
    
    Inputs
    ----------
    port_ret: DataFrame
        dataframe containing weekly portfolio returns, both overall and sector subportfolios
    port_risk: dataframe
        dataframe containing portfolio risk

    Outputs
    -------
    None
    
    """
    
    plt.figure(figsize=(10,6))
    plt.plot(2*port_risk.shift(periods = 1), color = 'b', label = '2\u03C3 Risk')
    plt.plot(-2*port_risk.shift(periods = 1), color = 'b', label = '')
    plt.plot(port_ret, '*', color = 'r', label = 'Return')
    plt.title('Weekly 2\u03C3 Risk Envelope Plot')
    plt.legend(loc = 'best')
    plt.show()

def plot_vol(f_vol, min_history = 52):
    """
    plot weekly factor volatilities 
    
    Inputs
    ----------
    f_vol: dataframe
        dataframe containing the weekly factor volatilities
    min_history: int, Default = 52
        minimum number of weeks to calculate volatility

    Outputs
    -------
    None

    """
    
    
    plt.figure(figsize=(12,6))
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, f_vol.shape[1]))))
    plt.plot(100*f_vol[min_history:])
    plt.legend(f_vol.columns)
    plt.title('Weekly Factor Volatilities (%)')    
    

def pull_price_data(d0, price_data):
    """
    pull price for date d0, with holiday handling logic to pull previous date 
    if date is missing

    Inputs
    ----------
    d0: datetime
        a single date, to pull price
    price_data: xarray
        xarray containing all pricing data 

    Outputs
    -------
    p0: xarray
        Description of return value

    """
    try:
        p0 = price_data.sel(date = d0, drop = True)[['Adj Close']]
    except:
        d1 = d0 - pd.Timedelta(1, unit='d')
        p0 = price_data.sel(date = d1, drop = True)[['Adj Close']]
    
    return p0



### pulls S&P 500 data from wikipedia site
def save_sp500_data():
    """
    Pull S&P500 information from wikipedia page. 

    Inputs
    ----------
    None
    
    Outputs
    -------
    SP_data: xarray
        xarray containing sector information pulled from wikipedia page

    """
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
        
        tickers.append(ticker.strip('\n')) # remove /n from end of tickers
        companies.append(company_name.strip('\n'))
        sectors.append(sector.strip('\n'))
        subsectors.append(subsector.strip('\n'))
        

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




def winsorize(ret, plot_scatter = False):
    """
    winsorize weekly factor returns, putting a cap and a floor on return outside mean +/- 5IQR. 
    
    Inputs
    ----------
    ret: xarray
        xarray containing 
    plot_scatter: bool, default False
        If True, plot scatterplot showing returns before/after winsorization

    Outputs
    -------
    output: xarray
        returns after winsorization

    """
    IQR = ret.quantile(0.75) - ret.quantile(0.25) #interquartile range
    IQR = IQR['Adj Close'].values
    ret_median = ret.quantile(0.5)['Adj Close'].values
    
    cap = ret_median + 5*IQR
    floor = ret_median - 5*IQR

    output = ret.clip(floor, cap)
    
    if plot_scatter == True:
        plt.figure(figsize = (6,6))
        plt.plot(ret['Adj Close'], output['Adj Close'], '*')
        plt.xlabel('Original Returns')
        plt.ylabel('Winsorized Returns')
        plt.title('Comparison of Winsorized Returns')
        plt.show()
    
    return output






