# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:37:20 2020

@author: -
"""


# Application imports
import pandas as pd
import os
import numpy as np
from pandas_datareader import data as pdr

companies = ['AAPL', 'ABBV','ABT','ACN','ADBE','BTC', 'AIG','ALL','AMGN','AMT','AMZN','AXP','BA','BAC',
             'BIIB','BK','BKNG','BLK','BMY','BRK.B','BRK', 'C','CAT','CHTR','CL','CMCS','COF','COP','COST',
             'CRM','CSCO','CVS','CVX','DD','DHR','DIS','DOW','DUK','EMR','EXC','F','FB','FDX',
             'GD','GE','GILD','GM','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KHC',
             'KMI','KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MRK','MS','MSFT',
             'NEE','NFLX','NKE','NVDA','ORCL','OXY','PEP','PFE','PG','PM','PYPL','QCOM','RTX','SBUX','TSLA',
             'SLB','SO','SPG','T','TGT','TMO','TXN','UNH','UNP','UPS','USB','V','VZ','WBA','WFC','WMT','XOM']


pwd = r"YourPathTodfNLPDailyScores"

df = pd.read_csv(os.path.join(pwd,"df_NLP_daily.csv"))
df['Date'] = df['Unnamed: 0']
df.index = pd.to_datetime(df.Date)
df = df.drop(['Date','Unnamed: 0'], axis=1)
df0 = df[df.index.dayofweek < 5]

prices_pwd = r"YourPathToDailyUSCompaniesPrices"

companies_not_available = list()
for company in companies:
    try:
        df_prices = pd.read_csv(os.path.join(prices_pwd,company+".csv"))
        df_prices = df_prices.dropna()
        df_prices.index = pd.to_datetime(df_prices.Date)
        df_prices = df_prices.reindex(df0.index)
        df_prices = df_prices.fillna(method='bfill')
        df0[company+'_Close'] = df_prices['Close']
    except FileNotFoundError:
        print(company)
        companies_not_available.append(company)

pos_scores_means = dict()
neg_scores_means = dict()

pos_scores_std = dict()
neg_scores_std = dict()

for company in companies:
    if company not in companies_not_available:
        pos_mean = np.mean(list(df0[company+'_dailyscore'][df0[company+'_dailyscore']>0.0]))
        neg_mean = np.mean(list(df0[company+'_dailyscore'][df0[company+'_dailyscore']<0.0]))
        pos_std = np.std(list(df0[company+'_dailyscore'][df0[company+'_dailyscore']>0.0]))
        neg_std = np.std(list(df0[company+'_dailyscore'][df0[company+'_dailyscore']<0.0]))
        
        pos_scores_means[company] = pos_mean
        neg_scores_means[company] = neg_mean
        pos_scores_std[company] = pos_std
        neg_scores_std[company] = neg_std
        
        print('Company {} has pos mean {} and neg mean {}'.format(company, pos_mean, neg_mean))
        print('')

for company in companies:
    if company not in companies_not_available:
        df0['Frets_'+company] = df0[company+'_Close'].pct_change().shift(-1)
        df0['pricerise_'+company] = np.where(df0['Frets_'+company]>=0,1,-1)

        posmean = pos_scores_means[company]
        posstd =  pos_scores_std[company]
        negmean = neg_scores_means[company]
        negstd =  neg_scores_std[company]
        
        df0['signal_'+company] = np.where(df0[company+'_dailyscore']>posmean+2*posstd,1,
                                  np.where(df0[company+'_dailyscore']<neg_scores_means[company]-3*negstd,1,0))

#SPY Data
start_date = "2020-05-17"
end_date = "2020-10-24"
data = pdr.get_data_yahoo("SPY", start_date, end_date)
data = data.dropna()
data = data.reindex(df0.index)
data = data.fillna(method='bfill')

df0['SPY_Close'] = data['Close']
df0['Frets_SPY'] = df0['SPY_Close'].pct_change().shift(-1)
df0['pricerise_SPY'] = np.where(df0['Frets_SPY']>=0,1,-1)

df1 = df0.dropna()
   
#Backtesting
for company in companies:
    if company not in companies_not_available:
        df1[company+'_strat_rets'] = df1['signal_'+company]*df1['Frets_'+company]

df1['Strategy Returns'] = 0.
for i in range(len(df1)):
    ret = 0
    count = 0
    for company in companies:
        if company not in companies_not_available:
            if df1[company+'_strat_rets'][i] != 0:
                ret+=df1[company+'_strat_rets'][i]
                count+=1
    if count!=0:
        ret/=count
    
    if float(ret)!=float(0):
        df1['Strategy Returns'][i] = ret

df1['Cumulative Strategy Returns'] = np.cumsum(df1['Strategy Returns'])
df1['Cumulative Market Returns'] = np.cumsum(df1['Frets_SPY'])
df1['Cumulative Strategy Returns'].plot() 

sharpe = np.sqrt(25)*np.mean(df1['Strategy Returns'])/np.std(df1['Strategy Returns'])

#Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(df1['Cumulative Strategy Returns'] , color='g', label='Strategy Returns')
plt.plot(df1['Cumulative Market Returns'], color='r', label='Market Returns')
plt.xlabel('Dates')
plt.ylabel('Returns')
plt.title('Strategy with a Sharpe Ratio of {}'.format(sharpe))
plt.legend()
plt.show()  
