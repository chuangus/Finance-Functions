# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 22:45:32 2020

@author: Angus
"""
import yfinance as yf
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt
import statsmodels.api as sm

def getdata(tickers, year):
    #Download data
    df = yf.download(tickers, period="max",interval='1mo')  ['Adj Close']
    #clean and calculate ln growth
    df = df.dropna(how='all')
    df2 = df.pct_change() + 1
    df2 = df2.dropna(how='all')
    df2 = np.log(df2)
    df2.index = pd.to_datetime(df2.index)
    df2 = df2[df2.index.year >= year]
    df2 = df2 * 12
    df2 = df2 [:-1]
    return df2

def rf(year):
    ### Data only from 1934
    sdt = dt.datetime(year, 1, 1)
    df = pdr.DataReader("TB3MS", "fred", sdt)
    df = df/100
    df = df.rename(columns={"TB3MS": "Ln Return"})
    return df

def sharpe(tickers, year):
    # year input >=1934
    df = getdata(tickers, year)
    df = df.reset_index(drop = False)
    df = df.set_index('Date')
    df2 = rf(year)
    #clean data
    Month_date = '-1-1'
    df = df.loc[str (str(year) + Month_date):df2.index[-1]]
    df2 = df2[-len(df):]
    #sharpe formula
    df3 = np.subtract(df,df2)
    df3 = df3.mean()
    df4 = df/12
    df4 = df4.std()*np.sqrt(12)
    Estimated_Sharpe_Ratio = df3/df4
    Estimated_Sharpe_Ratio = Estimated_Sharpe_Ratio.to_frame('Sharpe')
    print (Estimated_Sharpe_Ratio)
    
def bnchmark():
    ### data only from 1970 
    df = pd.read_excel('C:/Users/Angus/Downloads/historyIndex.xls', skiprows=6, index_col=0)
    df = df.dropna(how='all')
    df = df.replace(',','', regex=True)
    df = df.astype(float, errors = 'ignore')
    df2 = df.pct_change() + 1
    df2 = df2.dropna(how='all')
    df2 = np.log(df2) *12
    df2 = df2.rename(columns={"WORLD Standard (Large+Mid Cap) ": "Bnchmrk Return"})
    return df2

def alpbeta(tickers,year):
    ### input >= 1970
    df = pd.DataFrame()
    tickers = tickers.replace(' ','')
    tickers_list = tickers.split(',')
    for tickers in tickers_list:
        X = bnchmark()
        Y = getdata(tickers, year)
        #clean data
        month_date = '-1-1'
        Y = Y.loc[str (str(year) + month_date):X.index[-1]]
        X = X[-len(Y):]       
        X = np.array(X)
        Y = np.array(Y)
        ###Regression
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
        df2 = results.params      
        ### clean results
        df3 = pd.DataFrame({'Alpha': df2 [0],'Beta': df2[1]}, index = [tickers])
        df = df.append(df3)
    return df

from questrade_api import Questrade

# q = Questrade(refresh_token='a68s2Ml6cy-N-65Wk4XKfZ8sAUXQTd0r0')

q = Questrade()

###!!!###
account_number = 51750807
deposits = 1250
###!!!###


def questradestats():
    df = q.account_positions(account_number)
    df = df["positions"]
    df1 = pd.DataFrame()
    for df0  in df:
        #get symbol
        df0_symbol = df0['symbol']
        #get market value
        df0_currentMarketValue = df0['currentMarketValue']
        #find dividend
        ticker = yf.Ticker(df0_symbol)
        df = ticker.info
        df_previousClose = df['previousClose']
        df_yield = df['yield']
        df_dividend = df_yield*df_previousClose
        df_dividend = round(df_dividend,2)
        df_openQuantity = df0['openQuantity']
        df_dividend = df_dividend * df_openQuantity
        #clean data
        df = pd.DataFrame({'currentMarketValue': df0_currentMarketValue, 'dividend yield': "{:.2%}".format(df_yield), 'dividend': df_dividend}, index = [df0_symbol])
        df1 = df1.append(df)
    A = df1.sum(axis = 0)
    df0 = pd.DataFrame({'currentMarketValue': A[0], 'dividend yield': "{:.2%}".format(A[2]/A[0]), 'dividend': A[2]}, index = ['SUM'])
    df1 = df1.append(df0)
    return (df1)
    
def questrade_alpbeta():
    ## get symbols
    df = q.account_positions(account_number)
    df = df["positions"]
    y = ','
    for df0  in df:      
            df0_symbol = df0['symbol']
            x = df0_symbol
            y = x + y
    y = y[:-1]
    
    # get alpha beta
    df2 = alpbeta(y, 1950)
    df3 = questradestats() 
    # calculate weighted alp beta
    df3 = df3['currentMarketValue']
    df4 = df3.iloc[-1]
    df5 = df3/df4
    df5 = df5.iloc[:-1]
    df6 = df5.rename('Alpha')
    df7 = df5.rename ('Beta')
    df8 = pd.concat([df6, df7], axis=1)
    df9 = df2.mul(df8)
    return df9

def questrade_return():
    df = q.account_balances(account_number)
    df = df['combinedBalances']
    df = df[0]
    equity = df['totalEquity']
    print ("{:.2%}".format(equity/deposits-1),'total return since inception')

df = 'aapl'