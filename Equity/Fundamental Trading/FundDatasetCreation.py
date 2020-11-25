# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 09:51:21 2020

@author: ArmelFabrice
"""

import pandas as pd
import numpy as np
import os

import simfin as sf
from simfin.names import CLOSE, NET_INCOME, REVENUE,\
                         NET_PROFIT_MARGIN, SALES_GROWTH,\
                         ROA, ROE, TOTAL_EQUITY, TOTAL_ASSETS,\
                         REPORT_DATE, EARNINGS_GROWTH, SHARES_DILUTED,\
                         NET_INCOME_COMMON, FCF, NET_CASH_OPS, CAPEX, \
                         PSALES, PE, PFCF


# Set your API-key for downloading data.
sf.set_api_key('free')

# Set the local directory where data-files are stored.
# The dir will be created if it does not already exist.
sf.set_data_dir('C:/Users/user/Desktop/Systematic Trading/Strategies/MLStrategies/FundamentalDataPred/simfin_data/')

##Features
#Market Cap OK with df_prices
#Enterprise Value OK with df_prices and df_balance
#Trailing P/E OK with df_income and df_prices
#Forward P/E Impossible to have
#PEG Ratio OK with df_income and df_prices
#Price/Sales OK with df_prices and df_income
#Price/Book  OK with df_prices and df_balance
#Enterprise Value/Revenue OK with df_prices and df_balance and df_income
#Enterprise Value/EBITDA OK with df_prices and df_balance and df_income
#Profit Margin OK with df_income
#Operating Margin OK with df_income
#Return on Assets OK with df_income and df_balance
#Return on Equity OK with df_income and df_balance
#Revenue OK with df_income
#Revenue Per Share OK with df_income
#Qtrly Revenue Growth OK with df_income_qrt
#Gross Profit OK with df_income
#EBITDA OK with df_income and df_cashflow
#Net Income Avl to Common OK with df_income
#Diluted EPS OK with df_income
#Qtrly Earnings Growth OK with df_income_qrt
#Total Cash OK with df_balance
#Total Cash Per Share OK with df_balance
#Total Debt OK with df_balance
#Total Debt/Equity OK with df_balance
#Current Ratio OK with df_balance
#Book Value Per Share OK with df_prices and df_balance
#Operating Cash Flow OK with df_cashflow
#Levered Free Cash Flow OK with df_cashflow. Replaced by FCF
#Beta OK with df_prices
#50-days MA OK with df_prices
#200-days MA OK with df_prices
#Avg 3-months volume with df_prices
#Shares outstanding with df_prices
#Float = Number of shares publicly available to trade / Don't have access
#% Held by insiders Don't have access
#% Held by institutions Don't have access
# Shares Short (as of) Don't have access
#Short ratio Don't have access
#Short % of float Don't have access
#Shares Short (prior month) Don't have access

# Data for USA.
market = 'us'

# TTM Income Statements.
df_income_ttm = sf.load_income(variant='ttm', market=market)
#Contains: Revenue, Shares(Diluted), Revenue Per Share, Gross Profit, 
#          Net Income (Common)
#          Diluted Earnings Per Share(Diluted EPS) = Net Income (Common) / Shares Diluted
#          Operating Margin = Operating Income (Loss) * 100 / Revenue
#          Profit Margin = (Revenue - Cost of Revenue) * 100 / Revenue
#          Net Profit Margin = Net Income * 100 / Revenue

# Quarterly Income Statements.
df_income_qrt = sf.load_income(variant='quarterly', market=market)
#Contains: Qtrly Earnings Growth = amount by with this quarter earnings exceeds 
#                                  the same quarter earnings for past year

# TTM Balance Sheets.
df_balance_ttm = sf.load_balance(variant='ttm', market=market)
#Contains: Total Debt = Short Term Debt + Long Term Debt, 
#          Total Debt/Equity = Total Debt / Total Equity,
#          Total Cash = Cash, Cash Equivalents & Short Term Investments
#          Total Cash Per Share = Total Cash / Shares Diluted
#          Total Assets
#          Current Ratio = Total Current Assets / Total Current Liabilities


# TTM Cash-Flow Statements.
df_cashflow_ttm = sf.load_cashflow(variant='ttm', market=market)
#Contains: Operating Cash Flow = Net Cash from Operating Activities
#Contains: Free Clash Flow = Net Cash from Operating Activities + Change in Fixed Assets & Intangibles

# Daily Share-Prices.
df_prices = sf.load_shareprices(variant='daily', market=market)
#Contains: Market Cap = Shares Outstanding * Adj. Close, 
#          Enterprise value = Market Cap + Total Debt - Total Cash
#          Price To Book value = Close / Book value(Total Equity / Shares Outstanding)
#          Price / Earnings To Growth (PEG) Ratio = PE / TTM Earnings Growth

#Render it for ticker list
tickers = ['EMN','RRC','EBAY','ADM','HPQ','NVDA','F','AKAM','ADBE','CME','AMZN','QCOM',
 'WMT','CRM','DIS','CERN','AMAT','AMGN','ATVI','CAT','CHK','FDX','HSY','TWX','XOM',
 'JWN','FLIR','WHR','BIIB','HP','KO','TGT','MRK','COST','FFIV','NTAP','UNH','FCX','SEE',
 'NKE','D','MET','INTU','VZ','PG','VLO','JNPR','PEG','ROST','PDCO','NSC','MKC','OKE',
 'LEG','EQT','GPC','YUM','ITW','EMR','AVY','GLW','AIZ','GIS','FL','ACI','PXD','AMP','R','NOC',
 'NDAQ','MU','RHI','MAT','PRU','GWW','FLR','KIM','PEP','DO','NE','CTL','ETR','ACN','MAS','HD',
 'BLK','COP','ATI','PFE','MMM','GE','MCD','CSCO','ORCL','BMY','JNJ','UPS','AIG','APA','ABT',
 'LLY','AZO','HAL','GILD','ORLY','UTX','S','BSX','EL','WMB','IRM','NEM','AA','HON','A','DNR',
 'HRL','BXP','COG','EFX','STZ','UNP','MCK','HIG','NBL','ADP','CELG','PCAR','APD','STX','BEN',
 'SWK','ROP','LUV','MO','CINF','TJX','EXPE','LSI','XRAY','IBM','DD','MRO','K','DE','CVX','HAS',
 'VRSN','CL','KMB','PH','VFC','AVP','SHW','GD','LH','SYY','IFF','LEN','CMI','HST','JCI','ROK',
 'FISV','MOLX','OMC','CPB','MUR','OXY','NUE','CTAS','SYK','LM','SWN','WDC','TXT','CAH','CB',
 'CI','BWA','CAG','CMG','CNX','CSX','CVS','FLWS','FOSL','VTR','LOW','X','GPS','HOV','SBUX',
 'T','DOV','BA','BBY','MGM','AMD','KR','ECL','BDX','CMCSA','NBR','HUM','GES','VAR','RL','AEE',
 'AIV','MNST','NI','FMC','WEC','ALL','KMX','SNA','GRMN','JBL','AEP','SRE','INTC','TXN','MYL','RTN',
 'XRX','LRCX','CCL','AXP','SPG','DUK','JEC','DHR','EW','ED','EIX','PNR','DKS','CNP','HOG',
 'APH','PFG','DVA','MDT','DRI','PCG','EOG','MCHP','PLD','SJM','PNW','AFL','TIF','DLTR','ETN',
 'RSG','MMC','TAP','MAR','IR','DTE','KSS','JCP','NOV','DGX','FIS','VMC','MA','VNO','PPG','NWL',
 'DHI','PGR','TRV','NYX','SLB','ICE','FTI','GME','LNC','WYNN','FE','SRCL','WAT','AMT','OI','AN',
 'ANF','AVB','AGN','WY','CF','PWR','TSN','BIG','FSLR','HES','EXC','PKI','DVN','CBS','KSU','PSA',
 'MOS','IGT','NRG','M','LMT','GT','MSFT','FLS','KLAC','AEO','SO','XLNX','IP','PPL','BAX','NFLX','PRGO',
 'TMO','NEE','BLL','DLX','IPG','MSI','POM','GOOG','TDC','ABC','HRB','ALXN','ADI','RSH','AON','THC',
 'FTR','V','SIRI','WM','SAI','DDS','EQR','PM','CLF','IVZ','DAL','DG''CMS','GNW','NTRI','UNM',
 'COV','COL','CCI','AET','ADSK','BMC','CA','GM','PSX','ESRX','MON','EMC','DISCA','CHRW','TEL',
 'MCO','SNI','PBI','FB','XYL','LYB','SKS','XL','XEL','TRIP','CLX','SCG','EXPR','HOT','ABBV','TER',
 'HAR','DLPH','AAPL']
        
df_income_qrt = df_income_qrt.loc[tickers].copy()
df_income_ttm = df_income_ttm.loc[tickers].copy()
df_balance_ttm = df_balance_ttm.loc[tickers].copy()
df_cashflow_ttm = df_cashflow_ttm.loc[tickers].copy()
df_prices = df_prices.loc[tickers].copy()
df_prices = df_prices.drop(['SimFinId','Dividend'], axis=1)

dates = [str(df_prices.loc['A'].index[0])[:10],str(df_prices.loc['A'].index[-1])[:10]]

from pandas_datareader import data as pdr
data_SPY = pdr.get_data_yahoo('SPY', dates[0], dates[1])

close_SPY = data_SPY['Close']

dates2 = df_prices.index.get_level_values('Date')
df_prices['SPY'] = close_SPY.loc[dates2].values

# Plot the raw share-prices from the original DataFrame.
#df_prices.loc[tickers[0], CLOSE].plot()

def price_data_transfo(df):
    """
    Calculate ttm variation for a single stock.
    Use sf.apply() with this function for multiple stocks.
    
    :param df_prices:
        Pandas DataFrame with raw share-prices for a SINGLE stock.
    
    :return:
        Pandas DataFrame with price-signals.
    """
    
    # Create new DataFrame for the signals.
    # 1Y rolling correlation
    df["rets"] = df[CLOSE].pct_change()
    df["SPYrets"] = df["SPY"].pct_change()
    df["retssquare"] = df["rets"]**2
    df["SPYretssquare"] = df["SPYrets"]**2
    df["multrets"] = df["rets"]*df["SPYrets"]
    
    df["cov"] = df["multrets"].rolling(252).mean()
    
    df["vol"] = np.sqrt(df["retssquare"].rolling(252).mean())
    df["volSPY"] = np.sqrt(df["SPYretssquare"].rolling(252).mean())
    
    df["corr"] = df["cov"]/(df["vol"]*df["volSPY"])
    df["beta"] = df["corr"]*df["vol"]/df["volSPY"]
    
    # Setting the index improves performance.
    df['FYV'] = 100*df[CLOSE].pct_change(60).shift(-60)
    df['FYVSPY'] = 100*df['SPY'].pct_change(60).shift(-60)
    df['MA50'] = df[CLOSE].rolling(50).mean()
    df['MA200'] = df[CLOSE].rolling(200).mean()
    
    #Avg 3-months volume
    df['3-months Volume'] = df['Volume'].rolling(60).mean()
    
    #Market Capitalization
    df['Market Cap'] = df['Shares Outstanding']*df['Adj. Close']
    
    return df


# Calculate all the price-signals for MULTIPLE stocks.
df_prices_int = sf.apply(df=df_prices, func=price_data_transfo)
df_prices_f = df_prices_int[['Close', 'Adj. Close', '3-months Volume',
                             'Shares Outstanding', 'SPY', 'FYV',
                             'FYVSPY', 'MA50', 'MA200', 'beta',
                             'Market Cap']]

#Income statement metrics
# Data from Income Statements.
df1 = df_income_ttm.copy()[['Shares (Diluted)', 'Net Income', 'Revenue',
                     'Gross Profit', 'Operating Income (Loss)', 
                     'Cost of Revenue', 'Net Income (Common)']]

df2 = df_income_qrt.copy()[['Revenue', 'Net Income (Common)']]
df2['Revenue Qrt'] = df2['Revenue']
df2['Net Income (Common) Qrt'] = df2['Net Income (Common)']
df2 = df2.drop(['Revenue','Net Income (Common)'], axis=1)

# Combine the data into a single DataFrame.
df_join = pd.concat([df1, df2], axis=1)

#Contains: Revenue, Shares(Diluted), Revenue Per Share, Gross Profit, 
#          Net Income (Common)
#          Diluted Earnings Per Share(Diluted EPS) = Net Income (Common) / Shares Diluted
#          Operating Margin = Operating Income (Loss) * 100 / Revenue
#          Profit Margin = (Revenue - Cost of Revenue) * 100 / Revenue
#          Net Profit Margin = Net Income * 100 / Revenue

def income_data_transfo(df):
    """
    Calculate income statement metrics for a single stock.
    Use sf.apply() with this function for multiple stocks.
    
    :param df:
        Pandas DataFrame with required data from
        Income Statements, Balance Sheets, etc.
        Assumed to be TTM-data.
    
    :return:
        Pandas DataFrame with financial signals.
    """
    #Diluted EPS
    df['Revenue Per Share'] = df['Revenue'] / df['Shares (Diluted)']
    
    #Diluted EPS
    df['Diluted EPS'] = df['Net Income (Common)'] / df['Shares (Diluted)']
    
    #Profit Margin
    df['Profit Margin'] = 100 * (df['Revenue'] - df['Cost of Revenue']) / df['Revenue']
    
    #Operating Margin
    df['Operating Margin'] = 100 * df['Operating Income (Loss)'] / df['Revenue']
    
    # Net Profit Margin.
    df['Net Profit Margin'] = 100 * df[NET_INCOME] / df[REVENUE]
    
    # Quartely Revenue/Sales Growth
    df['Quarterly Revenue Growth'] = 100 * df['Revenue Qrt'] / df['Revenue Qrt'].shift(4) - 1
    
    # Quartely Earnings Growth
    df['Quarterly Earnings Growth'] = 100 * df['Net Income (Common) Qrt'] / df['Net Income (Common) Qrt'].shift(4) - 1
    
    #TTM Earnings Growth
    df['TTM Earnings Growth'] = 100 * df['Net Income (Common)'] / df['Net Income (Common)'].shift(4) - 1

    return df

df_income_f = sf.apply(df=df_join, func=income_data_transfo)

# Balance Sheet metrics
# Columns to keep
df_balance_reduced = df_balance_ttm[['Shares (Diluted)', 'Total Current Assets',
                                     'Cash, Cash Equivalents & Short Term Investments',
                                     'Total Assets', 'Total Current Liabilities',
                                     'Short Term Debt', 'Long Term Debt',
                                     'Total Equity']]
df_balance_reduced['Total Cash'] = df_balance_reduced['Cash, Cash Equivalents & Short Term Investments']

df_balance_reduced = df_balance_reduced.drop(['Cash, Cash Equivalents & Short Term Investments'], axis = 1)

#Contains: Total Debt = Short Term Debt + Long Term Debt, 
#          Total Debt/Equity = Total Debt / Total Equity,
#          Total Cash = Cash, Cash Equivalents & Short Term Investments
#          Total Cash Per Share = Total Cash / Shares Diluted
#          Total Assets
#          Current Ratio = Total Current Assets / Total Current Liabilities

def balance_data_transfo(df):
    """
    Calculate balance sheet metrics for a single stock.
    Use sf.apply() with this function for multiple stocks.
    
    :param df:
        Pandas DataFrame with required data from
        Income Statements, Balance Sheets, etc.
        Assumed to be TTM-data.
    
    :return:
        Pandas DataFrame with financial signals.
    """
    #Total Debt
    df['Total Debt'] = df['Short Term Debt'] + df['Long Term Debt']
    
    #Total Debt/Equity
    df['Total Debt/Equity'] = df['Total Debt'] / df['Total Equity']
    
    #Total Cash Per Share
    df['Total Cash Per Share'] = df['Total Cash'] / df['Shares (Diluted)']
    
    # Current Ratio.
    df['Current Ratio'] = df['Total Current Assets'] / df['Total Current Liabilities']
    
    return df

df_balance_f = sf.apply(df=df_balance_reduced, func=balance_data_transfo)\
                 .drop(['Shares (Diluted)'], axis=1)

#Cashflow data
df_cashflow_reduced = df_cashflow_ttm[['Net Cash from Operating Activities',
                                       'Change in Fixed Assets & Intangibles']]

df_cashflow_reduced['Operating Cash Flow'] = df_cashflow_reduced['Net Cash from Operating Activities']
df_cashflow_reduced['Free Cash Flow'] = df_cashflow_reduced['Net Cash from Operating Activities'] +\
                                        df_cashflow_reduced['Change in Fixed Assets & Intangibles']

df_cashflow_f = df_cashflow_reduced.drop(['Net Cash from Operating Activities',
                                          'Change in Fixed Assets & Intangibles'], axis = 1)

#Combined metrics calculus
df_all = pd.concat([df_income_f, df_balance_f, df_cashflow_f], axis=1)

def combined_data_calculus(df):
    """
    Calculate combined metrics for a single stock.
    Use sf.apply() with this function for multiple stocks.
    
    :param df:
        Pandas DataFrame with required data from
        Income Statements, Balance Sheets, etc.
        Assumed to be TTM-data.
    
    :return:
        Pandas DataFrame with financial signals.
    """
    #Return on Assets
    df['Return on Assets'] = 100 * df[NET_INCOME] / df[TOTAL_ASSETS].shift(4)
    
    #Return on Equity
    df['Return on Equity'] = 100 * df[NET_INCOME] / df[TOTAL_EQUITY].shift(4)
    
    
    return df

df_all_f = sf.apply(df=df_all, func=combined_data_calculus)

ebitda = sf.derived.ebitda(df_income_ttm, df_cashflow_ttm, formula='Net Income')

df_all_ff = pd.concat([df_all_f, ebitda], axis=1)
                       
# Reindex datasets to the same days as share-price data.
df_all_daily = sf.reindex(df_src=df_all_ff,
                          df_target=df_prices_f,
                          method='ffill')

df_prices_metrics = pd.concat([df_prices_f, df_all_daily], axis=1)

#Combined Data 2
def combined_data_calculus2(df):
    """
    Calculate combined metrics for all stocks in the DataFrames.
    
    :param df:
        Pandas DataFrame with share-prices, Income Statement,
        Balance Sheet and Cashflow TTM data for multiple stocks.
    
    :return:
        Pandas DataFrame with combined data.
    """

    # Enterprise value
    df['Enterprise Value'] = df['Market Cap'] + df['Total Debt'] - df['Total Cash']
    
    # Enterprise value Per Revenue
    df['Enterprise value Per Revenue'] = df['Enterprise Value'] / df['Revenue']
    
    # Enterprise value Per EBITDA
    df['Enterprise value Per EBITDA'] = df['Enterprise Value'] / df['EBITDA']
    
    #Earnings Per Share
    df['Earnings Per Share'] = df['Net Income (Common)'] / df['Shares (Diluted)']
    
    #Price-to-Earnings (Trailing) Ratio
    df['PE'] = df['Close'] / df['Earnings Per Share']
    
    #Book Value Per Share
    df['Book Value Per Share'] = df['Total Equity'] / df['Shares Outstanding']
    
    #Price-to-Book Ratio
    df['PB'] = df['Close'] / df['Book Value Per Share']
    
    #Price-to-Sales Ratio
    df['PS'] = df['Close'] / df['Revenue Per Share']
    
    #Price/Earnings-to-Growth Ratio
    df['PEG'] = (1/100) * df['PE'] / df['TTM Earnings Growth']

    return df

df_prices_metrics_f = sf.apply(df=df_prices_metrics, func=combined_data_calculus2)

#Quarterly dates
df_tmp = df_all_ff.copy()
df_tmp['E'] = df_tmp['EBITDA']
df_tmp = df_tmp['E']
df_tmp.index.set_names(['Ticker', 'Date'], inplace = True)

#Filter on quarterly dates
a = df_prices_metrics_f.join(df_tmp, on = ['Ticker', 'Date'], how = 'inner')
a2 = a.replace([np.inf, -np.inf], np.nan)
df_final = a2.dropna().drop(['E'], axis = 1)

df_final['Forward Semester Returns'] = df_final['FYV']
df_final['Forward SPY Semester Returns'] = df_final['FYVSPY']
df_final = df_final.drop(['FYV','FYVSPY'], axis = 1)

pwd = r"YourPath"
df_final.to_csv(os.path.join(pwd,"dataset.csv"))
