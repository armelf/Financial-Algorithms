# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:50:11 2019

@author: ArmelFabrice
"""

##### import the necessary modules and set chart style####
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.style.use('bmh')
import matplotlib.pylab as plt
import statsmodels.api as sm
from math import sqrt

import warnings
warnings.filterwarnings("ignore")

def dateparse2(s):
    string = s[:2]+'/' + s[3:5] + '/' + s[6:10] + ' '
    string +=s[11:16]
    return pd.datetime.strptime(string, '%d/%m/%Y %H:%M')

pwd = '../Data/'

# Ticker stock market prices
currs = ['EURUSD','AUDUSD','GBPUSD','NZDUSD','USDCHF','USDCAD','USDJPY',#Majors
         'AUDCAD','CADCHF','CADJPY','CHFJPY','EURAUD','EURCAD','EURCHF',
         'EURGBP','EURJPY','GBPCHF','GBPJPY','NZDJPY']

#0.00006 of commission
#Plus FXCM Transaction costs
trans_costs_list = [0.00021, 0.00025, 0.00027, 0.00029, 0.00024, 0.00032, 0.00022,
                    0.00038, 0.00043, 0.00033, 0.00036, 0.00038, 0.00039, 0.00033,
                    0.00033, 0.00029, 0.0004, 0.00037, 0.00032]

df0 = pd.DataFrame()
df00 = pd.DataFrame()

for ticker, trans_costs in zip(currs, trans_costs_list):
    path = pwd + '%s'%ticker+'1H'+'.csv'
    dftemp1 = pd.read_csv(path)
    dftemp1['Date'] = dftemp1['Local time']
    dftemp1['Date'] = dftemp1['Date'].apply(lambda x: dateparse2(x))
    dftemp1.index = dftemp1['Date']
    dftemp = dftemp1.resample('12H').agg({'Open': 'first', 'High': 'max', 'Low': 'min','Close': 'last', 'Volume': 'sum'})
    dftemp = dftemp[dftemp.index.dayofweek < 5]
    df0[ticker] = dftemp['Close']
    df00[ticker] = [trans_costs/2]*len(df0)

df00.index = df0.index
df0 = df0[df0.index.dayofweek < 5]
df00 = df00[df00.index.dayofweek < 5]

def func(comp, param, trans_costs, critical_level):

    dfa = df0[0:comp+param]
    dfb = df00[0:comp+param]
    
    #NOTE CRITICAL LEVEL HAS BEEN SET TO 5% FOR COINTEGRATION TEST
    
    def find_cointegrated_pairs(dataframe, critical_level = 0.05):
        n = dataframe.shape[1] # the length of dateframe
        pvalue_matrix = np.ones((n, n)) # initialize the matrix of p
        keys = dataframe.columns # get the column names
        pairs = [] # initilize the list for cointegration
        for i in range(n):
            for j in range(i+1, n): # for j bigger than i
                stock1 = dataframe[keys[i]] # obtain the price of "stock1"
                stock2 = dataframe[keys[j]]# obtain the price of "stock2"
                result = sm.tsa.stattools.coint(stock1, stock2) # get conintegration
                pvalue = result[1] # get the pvalue
                pvalue_matrix[i, j] = pvalue
                if pvalue < critical_level: # if p-value less than the critical level
                    pairs.append((keys[i], keys[j], pvalue)) # record the contract with that p-value
        return pvalue_matrix, pairs
    
    #set up the split point for our "training data" on which to perform the co-integration test (the remaining data will be fed to our backtest function)
    split = param
    
    #run our dataframe (up to the split point) of ticker price data through our co-integration function and store results
    pvalue_matrix,pairs = find_cointegrated_pairs(dfa[:-split], critical_level)
    
    def half_life(spread):
        spread_lag = spread.shift(1)
        spread_lag.iloc[0] = spread_lag.iloc[1]
        spread_ret = spread - spread_lag
        spread_ret.iloc[0] = spread_ret.iloc[1]
        spread_lag2 = sm.add_constant(spread_lag)
        model = sm.OLS(spread_ret,spread_lag2)
        res = model.fit()
        halflife = int(round(-np.log(2) / res.params[1],0))
    
        if halflife <= 0:
            halflife = 1
        return halflife
    
    
    def backtest(dfa, dfb, param, s1, s2, trans_costs=False):
        #############################################################
        # INPUT:
        # DataFrame of prices
        # s1: the symbol of contract one
        # s2: the symbol of contract two
        # x: the price series of contract one
        # y: the price series of contract two
        # OUTPUT:
        # df1['cum rets']: cumulative returns in pandas data frame
        # sharpe: Sharpe ratio
        # CAGR: Compound Annual Growth Rate
        
        #Kalman filter params
        delta = 1e-4
        wt = delta / (1 - delta) * np.eye(2)
        vt = 1e-3
        theta = np.zeros(2)
        C = np.zeros((2, 2))
        R = None
        
        # n = 4
        # s1 = pairs[n][0]
        # s2 = pairs[n][1]
        
        x = list(dfa[s1])
        y = list(dfa[s2])
        ts_x = list(dfb[s1])
        ts_y = list(dfb[s2])
        
        #X = sm.add_constant(x)
            
        # hrs = [0]*len(df)
        Fs = [0]*len(dfa)
        Rs = [0]*len(dfa)
        ets = [0]*len(dfa)
        Qts = [0]*len(dfa)
        sqrt_Qts = [0]*len(dfa)
        thetas = [0]*len(dfa)
        Ats = [0]*len(dfa)
        Cs = [0]*len(dfa)
        
        for i in range(len(dfa)-param-10, len(dfa)):
            # res = sm.OLS(y[i-split+1:i+1],X[i-split+1:i+1]).fit()
            # hr[i] = res.params[1]
            
            F = np.asarray([x[i], 1.0]).reshape((1, 2))
            Fs[i] = F
            if R is not None:
                R = C + wt
            else:
                R = np.zeros((2, 2))
            Rs[i] = R
        
            # Calculate the Kalman Filter update
            # ----------------------------------
            # Calculate prediction of new observation
            # as well as forecast error of that prediction
            
            yhat = F.dot(theta)
            et = y[i] - yhat
            ets[i] = et[0]
            
            # Q_t is the variance of the prediction of
            # observations and hence \sqrt{Q_t} is the
            # standard deviation of the predictions
            Qt = F.dot(R).dot(F.T) + vt
            sqrt_Qt = np.sqrt(Qt)[0][0]
            
            Qts[i] = Qt
            sqrt_Qts[i] = sqrt_Qt
            
            # The posterior value of the states \theta_t is
            # distributed as a multivariate Gaussian with mean
            # m_t and variance-covariance C_t
            At = R.dot(F.T) / Qt
            theta = theta + At.flatten() * et
            C = R - At * F.dot(R)
            
            thetas[i] = theta[0]
            Ats[i] = At
            Cs[i] = C
            
        dfa['et'] = ets
        # df['et'] = df['et'].rolling(5).mean()
        dfa['sqrt_Qt'] = sqrt_Qts
        dfa['theta'] = thetas
        
        # run regression (including Kalman Filter) to find hedge ratio and then create spread series
        df1 = pd.DataFrame({'y':y,'x':x, 'ts_x':ts_x, 'ts_y':ts_y, 'et':dfa['et'], 'sqrt_Qt':dfa['sqrt_Qt'], 'theta':dfa['theta']})[-param:]
        
        # df1[['et','sqrt_Qt']].plot()
        
        # df1['spread0'] = df1['y'] - df1['theta']*df1['x']
    
        # # calculate half life
        # halflife = half_life(df1['spread0'])
    
        # # calculate z-score with window = half life period
        # meanSpread = df1.spread0.rolling(window=halflife).mean()
        # stdSpread = df1.spread0.rolling(window=halflife).std()
        # df1['zScore'] = (df1.spread0-meanSpread)/stdSpread
    
        # #############################################################
        # # Trading logic
        # entryZscore = 0.8
        # exitZscore = 0.2
    
        #set up num units long
        # df1['long entry'] = ((df1.zScore < - entryZscore) & (df1.zScore.shift(1) > - entryZscore))
        # df1['long exit'] = ((df1.zScore > - exitZscore) & (df1.zScore.shift(1) < - exitZscore))
        
        df1['long entry'] = ((df1.et < -0.002) & (df1.et.shift(1) > -0.002))
        df1['long exit'] = ((df1.et > -0.000) & (df1.et.shift(1) < -0.000))
        
        df1['num units long'] = np.nan 
        df1.loc[df1['long entry'],'num units long'] = 1 
        df1.loc[df1['long exit'],'num units long'] = 0 
        df1['num units long'][0] = 0 
        df1['num units long'] = df1['num units long'].fillna(method='pad') 
        
        #set up num units short 
        # df1['short entry'] = ((df1.zScore > entryZscore) & (df1.zScore.shift(1) < entryZscore))
        # df1['short exit'] = ((df1.zScore < exitZscore) & (df1.zScore.shift(1) > exitZscore))
        
        df1['short entry'] = ((df1.et > 0.002) & (df1.et.shift(1) < 0.002))
        df1['short exit'] = ((df1.et < 0.000) & (df1.et.shift(1) > 0.000))
        
        df1.loc[df1['short entry'],'num units short'] = -1
        df1.loc[df1['short exit'],'num units short'] = 0
        df1['num units short'][0] = 0
        df1['num units short'] = df1['num units short'].fillna(method='pad')
    
        df1['numUnits'] = df1['num units long'] + df1['num units short']
        df1['signals'] = df1['numUnits'].diff()
        #df1['signals'].iloc[0] = df1['numUnits'].iloc[0]
        
        df1['yfrets'] = df1['y'].pct_change().shift(-1)
        df1['xfrets'] = df1['x'].pct_change().shift(-1)
        
        if trans_costs == True:
            df1['spread'] = (df1['y']*(1+df1['signals']*df1['ts_y'])) - (df1['theta']*(df1['x']*(1-df1['signals']*df1['ts_x'])))
            #df1['spread'] = (df1['y']*(1+df1['signals']*0.0001))
        else:
            df1['spread'] = df1['y'] - df1['theta']*df1['x']
            #df1['spread'] = df1['y']
            
        df1['spread pct ch'] = (df1['spread'] - df1['spread'].shift(1)) / ((df1['x'] * abs(df1['theta'])) + df1['y'])
        #df1['spread pct ch'] = (df1['spread'] - df1['spread'].shift(1)) / df1['spread'].shift(1)
        df1['port rets'] = df1['spread pct ch']*df1['numUnits'].shift(1)
    
        df1['cum rets'] = df1['port rets'].cumsum()
        df1['cum rets'] = df1['cum rets'] + 1
        
        #df1 = df1.dropna()
        ##############################################################
    
        try:
            sharpe = ((df1['port rets'].mean() / df1['port rets'].std()) * sqrt(252*2))
        except ZeroDivisionError:
            sharpe = 0.0
    
        ##############################################################
        start_val = 1
        end_val = df1['cum rets'].iloc[-1]
        
        # print(len(df1[df1['long entry']==True])+len(df1[df1['short entry']==True]))
        # print(end_val)
    
        start_date = df1.iloc[0].name
        end_date = df1.iloc[-1].name
    
        days = (end_date - start_date).days
    
        CAGR = round(((float(end_val) / float(start_val)) ** (252.0/days)) - 1,4)
    
        df1[s1+ " "+s2] = df1['cum rets']
    
        return df1[s1+" "+s2], sharpe, CAGR
    
    results = []
        
    for pair in pairs:
        
        rets, sharpe,  CAGR = backtest(dfa, dfb, param, pair[0], pair[1], trans_costs)
        results.append(rets)
        #print("The pair {} and {} produced a Sharpe Ratio of {} and a CAGR of {}".format(pair[0],pair[1],round(sharpe,2),round(CAGR,4)))
        #rets.plot(figsize=(20,15),legend=True)
    
    #concatenate together the individual equity curves into a single DataFrame
    try:
        results_df = pd.concat(results,axis=1).dropna()
        
        #equally weight each equity curve by dividing each by the number of pairs held in the DataFrame
        results_df /= len(results_df.columns)
        
        #sum up the equally weighted equity curves to get our final equity curve
        final_res = results_df.sum(axis=1)
        
        #plot the chart of our final equity curve
        plt.figure()
        final_res.plot(figsize=(20,15))
        plt.title('Between {} and {}'.format(str(results_df.index[0])[:10],str(results_df.index[-1])[:10]))
        plt.xlabel('Date')
        plt.ylabel('Returns')
        
        #calculate and print our some final stats for our combined equity curve
        try:
            sharpe = (final_res.pct_change().mean() / final_res.pct_change().std()) * (sqrt(252*2))
        except ZeroDivisionError:
            sharpe = 0.0
        start_val = 1
        end_val = final_res.iloc[-1]
        
        start_date = final_res.index[0]
        end_date = final_res.index[-1]
        
        days = (end_date - start_date).days
        
        CAGR = round(((float(end_val) / float(start_val)) ** (252.0/days)) - 1,4)
        print("Sharpe Ratio is {} and CAGR is {}".format(round(sharpe,2),round(CAGR,4)))
    except ValueError:
       # return "No result"
        print('No pair found')

indexes = df0.index

param = 500
compini = 1000
comps = [i for i in range(compini,len(df0),param)]
for comp in comps:
    print('')
    print('Between {} and {}'.format(str(indexes[comp])[:10],str(indexes[min(comp+param,len(indexes)-1)])[:10]))
    func(comp, param, trans_costs = True, critical_level = 0.2)
