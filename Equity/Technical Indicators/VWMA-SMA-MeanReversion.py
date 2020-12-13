# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:51:21 2019

@author: ArmelFabrice
"""

import pandas as pd
import numpy as np
import ta_strategies

import warnings
warnings.filterwarnings('ignore')

# Save data to this files
pwd = '../Data/'

def take_profit_stop_loss(freq):
    if freq == 'd':
        tp = 0.0098
        sl = -tp/3
    else:
        tp = 0.0028
        sl = -tp/3
    return tp, sl  
        
def create_df(freq):

    df = pd.read_csv(pwd+'SPY'+freq+'.csv')
       
    df.index = df['Date']
    df = df[['Low','High','Close','Open','Volume']]
    
    # Double check the result
    df.head()
    
    #Take profit/stop-loss
    tpsl = take_profit_stop_loss(freq)
    sl = tpsl[1]
    
    #Trend
    df['Trend'] = ['']*len(df)
    df['NTrend'] = [0]*len(df)
    
    df['MA150'] = df['Close'].rolling(window=150).mean()
    ma = list(df['MA150'].pct_change())
    mas = list(df['MA150'].pct_change().rolling(150).std())
    
    for i in range(25,len(df)-1):
        if ma[i]>=mas[i]:
            df['Trend'][i] = 'Uptrend'
            df['NTrend'][i+1] = 1
        elif ma[i]<=-mas[i]:
            df['Trend'][i] = 'Downtrend'
            df['NTrend'][i+1] = -1
        else:
            df['Trend'][i] = 'Range'
            df['NTrend'][i+1] = 0
    
    #Signals creation
    df = ta_strategies.vwsma_strategy(df, sl)
    df['Trend_signal'] = df['NTrend']
    
    #Price Rise
    df['Price Rise'] = np.where(df['Close'].shift(-1) >= df['Close'], 1, -1)
        
    #Interpolate
    df1 = df.dropna()
    df1 = df1.interpolate()

    df1 = df1[['VWSMA_signal', 'Close']]
    
    return df1

freq = 'd'

df = create_df(freq)

def test_factor_acc(df, factor):
    sharpe = 0
    best = 0
    
    for n in range(1,10):
    
        df1 = df.dropna()
        df1 = df1.interpolate()
        trade_df = pd.DataFrame()
        trade_df['y_pred'] = df1[factor].rolling(window=n).mean()
        trade_df['Close'] = df1['Close']
        trade_df = trade_df.dropna()
        
        #Computing strategy returns                                        
        
        trade_df['Tomorrows Returns'] = 0.
        trade_df['Tomorrows Returns'] = np.log(trade_df['Close']/trade_df['Close'].shift(1))
        trade_df['Tomorrows Returns'] = trade_df['Tomorrows Returns'].shift(-1)
        
        trade_df['Strategy Returns'] = 0.
        for i in range(len(trade_df)):
            x = trade_df['y_pred'][i]
            if x > 0:
                trade_df['Strategy Returns'][i] = trade_df['Tomorrows Returns'][i]
            elif x == 0:
                trade_df['Strategy Returns'][i] = 0
            elif x < 0:
                trade_df['Strategy Returns'][i] = -trade_df['Tomorrows Returns'][i]
            
        #Cumulative returns
        trade_df['Cumulative Market Returns'] = np.cumsum(trade_df['Tomorrows Returns'])
        trade_df['Cumulative Strategy Returns'] = np.cumsum(trade_df['Strategy Returns'])
        
        sharpe1 = np.sqrt(252)*np.mean(trade_df['Strategy Returns'])/np.std(trade_df['Strategy Returns'])
        
        if sharpe1>sharpe:
            sharpe = sharpe1
            best = n
            cummktrets = trade_df['Cumulative Market Returns']
            cumstratrets = trade_df['Cumulative Strategy Returns']
            
    print('Best sharpe {} for n = {}'.format(sharpe, best))
    
    return sharpe, best, cummktrets, cumstratrets

def plot_equity_curve(cummktrets, cumstratrets):
#    Plot
    import matplotlib.pyplot as plt
    
    x = np.array(cummktrets)
    y = np.array(cumstratrets)
    
    plt.figure(figsize=(10,5))
    plt.plot(x, color='r', label='SPY Market Returns')
    plt.plot(y, color='g', label='Strategy Returns')
    plt.title('Strategy Returns Without Reinvestment Between 1993 and 2019')
    plt.legend()
    plt.show()

#Equity curve plot   
test_acc = test_factor_acc(df, 'VWSMA_signal')    
cummktrets = test_acc[2]
cumstratrets = test_acc[3]

plot_equity_curve(cummktrets, cumstratrets)
