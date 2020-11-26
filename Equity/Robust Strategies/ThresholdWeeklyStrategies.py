# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 03:34:44 2020

@author: ArmelFabrice
"""

import simfin as sf
import pandas as pd
from datetime import datetime
import numpy as np
import ta

# Set your API-key for downloading data.
sf.set_api_key('free')

# Set the local directory where data-files are stored.
# The dir will be created if it does not already exist.
sf.set_data_dir('DirPath)

# Data for USA.
market = 'us'

# Daily Share-Prices.
df_prices = sf.load_shareprices(variant='daily', market=market)
        
#Obtain SP500 tickers
import urllib.request
from html_table_parser import HTMLTableParser

url_snp500 = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

def obtain_parse_wiki_stocks_sp500(url):
  """Download and parse the Wikipedia list of S&P500 
  constituents using requests and libxml.

  Returns a list of tuples for to add to MySQL."""

  # Get S&P500 website content
  req = urllib.request.Request(url)
  response = urllib.request.urlopen(req)
  data = response.read().decode('utf-8')
  
  #Instantiate the parser and feed it
  p = HTMLTableParser()
  p.feed(data)
  table_list = p.tables
  table = table_list[0][1:]

  # Obtain the symbol information for each row in the S&P500 constituent table
  symbols = []
  for row in table:
    sd = {'ticker': row[0],
        'name': row[1],
        'sector': row[3]}
    # Create a tuple (for the DB format) and append to the grand list
    symbols.append(sd['ticker'])
  return symbols

ref = 'AAPL'
tickers = obtain_parse_wiki_stocks_sp500(url_snp500)
#tickers = [ref,tickers[0], tickers[1], tickers[2]]
#tickers = [ref]

#Daily Prices
df_prices_daily = df_prices.loc[tickers].copy()

#Weekly Prices
def convert_daily_weekly(df_prices_daily):
    
    df_weekly = df_prices_daily.resample('W',loffset=pd.offsets.timedelta(days=-6)).agg({'Open': 'first', 'High': 'max', 'Low': 'min','Close': 'last', 'Volume': 'sum'}) 
    return df_weekly

df_prices = sf.apply(df=df_prices_daily, func=convert_daily_weekly)
    
print(df_prices.head())


#Calculate wanted returns and technical indicators
def calc(df):
    # Create new DataFrame for the signals.
    # Setting the index improves performance.
    df_rets_ta = pd.DataFrame(index=df.index)
    
    # Close
    df_rets_ta['Close'] = df['Close']
    
    # High
    df_rets_ta['High'] = df['High']
    
    # Close returns
    df_rets_ta['Crets'] = df['Close'].pct_change()

    # Forward Close Close returns
    df_rets_ta['FCCrets'] = pd.Series((df['Close'].shift(-1)-df['Close'])/df['Close'])
        
    # High Close returns
    df_rets_ta['HCrets'] = pd.Series((df['High']-df['Close'])/df['Close'])
    
    # Forward High Close returns
    df_rets_ta['FHCrets'] = pd.Series((df['High'].shift(-1)-df['Close'])/df['Close'])
    
    #Forward High High returns
    df_rets_ta['FHHrets'] = pd.Series((df['High'].shift(-1)-df['High'])/df['High'])
    
    #Commodity Channel Index
    df_rets_ta['CCI'] = ta.trend.cci(df['High'],df['Low'], df['Close'], n=20, c=0.015)
    
    #Simple Moving Average 10-days
    df_rets_ta['MA10'] = df['Close'].rolling(window=10).mean()
    
    #Exponential Moving Average 10 days
    df_rets_ta['EMA10'] = df['Close'].ewm(span=10).mean()
    
    #Exponential Moving Average 20 days
    df_rets_ta['EMA20'] = df['Close'].ewm(span=20).mean()
    
    #Zero Lag exponential 10-day moving average
    df_rets_ta['ZLEMA10'] = 0.
    n_days = 10
    a = 2/(n_days+1)
    b = int((n_days-1)/2)
    df_rets_ta['ZLEMA10'] = a*(2*df_rets_ta['Close']-df_rets_ta['Close'].shift(b)) + (1-a*df_rets_ta['ZLEMA10'].shift(1))
    
    #Weighted 10-days moving average
    df_rets_ta['WMA10'] = 0.
    n_days = 10
    sumdays = np.sum(1+np.arange(n_days))
    for d in range(n_days):
        df_rets_ta['WMA10']+=(d+1)*df_rets_ta['Close'].shift(n_days-d)
    df_rets_ta['WMA10']/=sumdays
    
    #Relative Strength Index
    df_rets_ta['RSI'] = ta.momentum.rsi(df['Close'], n=14)
    
    #Momentum 10 days
    df_rets_ta['MOM10'] = df['Close']-df['Close'].shift(10)
    
    #Rate of Change 10 days
    df_rets_ta['ROC10'] = 100*(df['Close']-df['Close'].shift(10))/df['Close'].shift(10)
    
    #SAR
    df_rets_ta['SAR'] = ta.trend.sar(df,af=0.02, amax=0.2)
    
    #Volume
    df_rets_ta['Volume'] = df['Volume']
    
    #Force Index
    days = 1
    df_rets_ta['FI'] = df['Close'].diff(days) * df['Volume']
    
    #High Rise
    df_rets_ta['HR'] = pd.Series(np.where(df['High'].shift(-1) >= df['High'], 1, 0),index=df.index)
    
    return df_rets_ta.dropna()

def rets_ta_tickers(df_prices):
    df_rets_ta = sf.apply(df=df_prices, func=calc)
    df_rets_ta = df_rets_ta.dropna()
    
    #Available tickers
    mi=df_rets_ta.index
    all_indexes=list(mi.get_level_values(0))
    ticker_ref = ref
    df_ret_ta = df_rets_ta.loc[ticker_ref]
    indexes = df_ret_ta.index
    tickersa = []
    for ticker in all_indexes:
        if ticker not in tickersa:
            df_ret_ta1 = df_rets_ta.loc[ticker]
            if list(df_ret_ta1.index) == list(indexes):
                tickersa.append(ticker)
    return df_rets_ta, tickersa

rets_cols = ['Crets', 'FCCrets', 'HCrets', 'FHCrets', 'FHHrets']

ta_cols = ['CCI', 'MA10', 'EMA10', 'EMA20', 'ZLEMA10', 'WMA10',\
           'RSI', 'MOM10', 'ROC10', 'SAR', 'Volume','FI']

df_rets_ta, tickersa = rets_ta_tickers(df_prices)
print(df_rets_ta.head())

df_rets = df_rets_ta[rets_cols]
df_ta = df_rets_ta[ta_cols]
highrise = df_rets_ta['HR']

def computeCAGR(df_rets, tickersa, stock_preds_dict, threshold, threshold_dict, tcosts = False, mode = 'test', naive=True):
    #Backtesting
    threshold = threshold
    threshold2 = threshold_dict[threshold]
    
    #Compute matrix confusion parameters when the D threshold is crossed
    TP, FP, TN, FN = (0, 0, 0, 0)
    
    #First ticker
    dfret = df_rets.loc[tickersa[0]]
    indexes = dfret.index
    
    trade_df = pd.DataFrame(index=dfret.index, columns=['FHHrets'+tickersa[0],'FHHSignal'+tickersa[0]])
    
    #Computing tickers 0 wanted returns
    trade_df['FHHrets'+tickersa[0]] = dfret['FHHrets'][indexes]
    
    trade_df['FHCrets'+tickersa[0]] = dfret['FHCrets'][indexes]
    trade_df['HCrets'+tickersa[0]] = dfret['HCrets'][indexes]
    
    trade_df['FCCrets'+tickersa[0]] = dfret['FCCrets'][indexes]
    trade_df['HCretsMA'+tickersa[0]] = dfret['HCrets'].shift(1).rolling(1).mean()[indexes]
    trade_df['FHCrets'+tickersa[0]] = dfret['FHCrets'][indexes]
    
    trade_df['FHHSignal'+tickersa[0]] = np.where(dfret['FHHrets']>=0,1,0)

    
    #Predictions for the first ticker
    trade_df['PredSignal'+tickersa[0]] = stock_preds_dict[str(mode)+'pred'+tickersa[0]]
    
    #Stock Strat returns
    trade_df['Stratrets'+tickersa[0]] = 0.
    
    for i in range(len(trade_df)):
        if trade_df['HCrets'+tickersa[0]][i] >= threshold and trade_df['HCretsMA'+tickersa[0]][i] >= threshold2:
            #If we predict high Rise
            if not naive:
                if trade_df['PredSignal'+tickersa[0]][i] == 1:
                    if trade_df['FHHSignal'+tickersa[0]][i]  == 1:
                        trade_df.loc[indexes[i],'Stratrets'+tickersa[0]] = trade_df['HCrets'+tickersa[0]][i]
                        TP +=1
                    else:
                        trade_df.loc[indexes[i],'Stratrets'+tickersa[0]] = trade_df['FCCrets'+tickersa[0]][i]
                        FP +=1
                else:
                    if trade_df['FHHSignal'+tickersa[0]][i]  == 1:
                        FN += 1
                    else:
                        TN +=1
            else:
                if trade_df['FHHSignal'+tickersa[0]][i]  == 1:
                    trade_df.loc[indexes[i],'Stratrets'+tickersa[0]] = trade_df['HCrets'+tickersa[0]][i]
                    TP +=1
                else:
                    trade_df.loc[indexes[i],'Stratrets'+tickersa[0]] = trade_df['FCCrets'+tickersa[0]][i]
                    FP +=1
                
    
    #Other tickers
    tickersaa = [tickersa[0]]
    for ticker in tickersa:
        if ticker!=tickersa[0]:
            dfret1 = df_rets.loc[ticker].copy()
            if list(dfret1.index) == list(indexes):
                tickersaa.append(ticker)
                #Computing ticker wanted returns
                trade_df['FHHrets'+ticker] = dfret1['FHHrets'][indexes]
                
                trade_df['FHCrets'+ticker] = dfret1['FHCrets'][indexes]
                trade_df['HCrets'+ticker] = dfret1['HCrets'][indexes]
                
                trade_df['FCCrets'+ticker] = dfret1['FCCrets'][indexes]
                trade_df['HCretsMA'+ticker] = dfret1['HCrets'].shift(1).rolling(1).mean()[indexes]
                trade_df['FHCrets'+ticker] = dfret1['FHCrets'][indexes]
                
                trade_df['FHHSignal'+ticker] = np.where(dfret1['FHHrets']>=0,1,0)
                
                #Predictions for the first ticker
                trade_df['PredSignal'+ticker] = stock_preds_dict[str(mode)+'pred'+ticker]
                
                #Stock Strat returns
                trade_df['Stratrets'+ticker] = 0.
                
                for i in range(len(trade_df)):
                    if trade_df['HCrets'+ticker][i] >= threshold and trade_df['HCretsMA'+ticker][i] >= threshold2:
                        #If we predict high Rise
                        if not naive:
                            if trade_df['PredSignal'+ticker][i] == 1:
                                if trade_df['FHHSignal'+ticker][i]  == 1:
                                    trade_df.loc[indexes[i],'Stratrets'+ticker] = trade_df['HCrets'+ticker][i]
                                    TP += 1
                                else:
                                    trade_df.loc[indexes[i],'Stratrets'+ticker] = trade_df['FCCrets'+ticker][i]
                                    FP += 1
                            else:
                                if trade_df['FHHSignal'+ticker][i]  == 1:
                                    FN += 1
                                else:
                                    TN +=1
                        else:
                            if trade_df['FHHSignal'+ticker][i]  == 1:
                                trade_df.loc[indexes[i],'Stratrets'+ticker] = trade_df['HCrets'+ticker][i]
                                TP += 1
                            else:
                                trade_df.loc[indexes[i],'Stratrets'+ticker] = trade_df['FCCrets'+ticker][i]
                                FP += 1
      
    trade_df['Strategy Returns'] = 0.
    for i in range(len(trade_df)):
        ret = 0
        count = 0
        for ticker in tickersaa:
            if trade_df['Stratrets'+ticker][i] != 0:
                ret+=trade_df['Stratrets'+ticker][i]
                count+=1
        if count!=0:
            ret/=count
        
        if float(ret)!=float(0):
            if tcosts == True:
                trade_df.loc[indexes[i],'Strategy Returns'] = ret - 0.0035
            else:
                trade_df.loc[indexes[i],'Strategy Returns'] = ret
        
    #Cumulative returns
    trade_df['Cumulative Strategy Returns'] = np.cumsum(trade_df['Strategy Returns'])
    
    #Sum of returns
    
    #Sharpe Ratio
    sharpe = np.sqrt(52)*np.mean(trade_df['Strategy Returns'])/np.std(trade_df['Strategy Returns'])
    
    #CAGR
    start_val = 1
    end_val = trade_df['Cumulative Strategy Returns'][-1]+1
    start_date = trade_df.index[0]
    end_date = trade_df.index[-1]
    trade_df.index = pd.to_datetime(trade_df.index)
    days = (end_date - start_date).days
    CAGR = (float(end_val) / float(start_val)) ** (252.0/days) - 1
    
    #Compute total D accuracy, precision and recall
    try:
        accuracyD = (TP + TN) / (TP + TN + FP + FN)
    except ZeroDivisionError:
        accuracyD = 0
    try:
        precisionD = TP / (TP + FP)
    except ZeroDivisionError:
        precisionD = 0
    try:
        recallD = TP / (TP + FN)
    except ZeroDivisionError:
        recallD = 0

    return trade_df, sharpe, CAGR, accuracyD, precisionD, recallD

def plot(returns,trade_df,df_spy):
    #Integrate SPY returns
    df_spy = df_spy[trade_df.index]
    trade_df['Close'] = df_spy['Close']
    trade_df = trade_df.dropna()
    
    #Computing Tomorrow SPY returns                                        
    
    trade_df['Tomorrow Benchmark Returns'] = 0.
    trade_df['Tomorrow Benchmark Returns'] = np.log(trade_df['Close']/trade_df['Close'].shift(1))
    trade_df['Tomorrow Benchmark Returns'] = trade_df['Tomorrow Benchmark Returns'].shift(-1)
    
    #Cumulative benchmark Returns
    trade_df['Cumulative Benchmark Returns'] = np.cumsum(trade_df['Tomorrow Benchmark Returns'])
    
    #Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(list(trade_df['Cumulative Benchmark Returns']), color='r', label='Market Returns')
    plt.plot(list(trade_df['Cumulative Strategy Returns']), color='g', label='Strategy Returns')
    plt.legend()
    plt.show()

#Select dates from a MultiIndex dataset
def selectdates(df, date1, date2):
    dateindex = df.index.get_level_values('Date')
    dateindex = pd.DatetimeIndex(dateindex)
    df1 = df.loc[dateindex >= date1]
    
    dateindex1 = df1.index.get_level_values('Date')
    dateindex1 = pd.DatetimeIndex(dateindex1)
    df2 = df1.loc[dateindex1 <= date2]
    return df2

def selectdates2(df, date):
    dateindex = df.index.get_level_values('Date')
    dateindex = pd.DatetimeIndex(dateindex)
    df1 = df.loc[dateindex >= date]
    return df1

#Build a SVM Classifying Model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')

from datetime import timedelta

#Create train-test bundles
def create_traintest_bundles(df_ta, ref, total_weeks, test_weeks):
    length = total_weeks
    delta = test_weeks
    df_ta_2 = df_ta.loc[ref]
    dates = list(df_ta_2.index)
    list_timeranges = []
    i = 0
    while i <= len(df_ta_2)-length*8/13:
        interval = list()
        date = str(dates[i])[:10]
        interval.append(date)
        
        #Compute the next date
        ddate = datetime.strptime(date, "%Y-%m-%d")
        dnextdate = ddate + timedelta(days=length*5)
        nextdate = dnextdate.strftime("%Y-%m-%d")
        
        interval.append(nextdate)
        list_timeranges.append(interval)
        i+=delta
    return list_timeranges
    
#Testing period
#df_rets_cc = selectdates2(df_rets, date2)
#df_ta_cc = selectdates2(df_ta, date2)
#highrise_cc = selectdates2(highrise, date2)

def traintest(df_ta, df_rets, tickersa, highrise, ratio, timeranges, tcosts = False, sc=False):
    threshold_list = [0., 0.04, 0.06, 0.09, 0.12, 0.15, 0.2]
    
    #Supporting Threshold Dict creation
    threshold_dict = dict()
    threshold_dict[0.] = 0.
    threshold_dict[round(0.04,2)] = 0.0
    threshold_dict[round(0.05,2)] = 0.0
    threshold_dict[round(0.06,2)] = 0.0
    threshold_dict[round(0.09,2)] = 0.0
    threshold_dict[round(0.12,2)] = 0.0
    threshold_dict[round(0.15,2)] = 0.0
    threshold_dict[round(0.2,2)] = 0.0
    
    timerange_stock_preds_dict = dict()
    threshold_repartition_dict = dict()
    
    for threshold in threshold_list:
        threshold_repartition_dict[threshold] = 0
    
    for timerange in timeranges:
        print('')
        print('Start timerange: {} to {}'.format(str(timerange[0]), str(timerange[1])))
        print('')
        
        tmp_stock_preds_dict = dict()
        
        avg_stock_acc_train_tmp = 0
        avg_stock_prec_train_tmp = 0
        avg_stock_rec_train_tmp = 0
        avg_stock_acc_test_tmp = 0
        avg_stock_prec_test_tmp = 0
        avg_stock_rec_test_tmp = 0
        
        list_stock_acc_train_tmp = list()
        list_stock_prec_train_tmp = list()
        list_stock_rec_train_tmp = list()
        list_stock_acc_test_tmp = list()
        list_stock_prec_test_tmp = list()
        list_stock_rec_test_tmp = list()
        
        for ticker in tickersa:
            df_ta_f = selectdates(df_ta, timerange[0], timerange[1])
            highrise_f = selectdates(highrise, timerange[0], timerange[1])
            X = df_ta_f.loc[ticker].copy()
            y = highrise_f.loc[ticker].copy()
            
            l = int((1-ratio)*len(X))
            
            X_train, X_test, y_train, y_test = X[:l], X[l:], y[:l], y[l:]
            
            train_indexes = list(y_train.index)
            test_indexes = list(y_test.index)
            
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)    
            
            if sc == True:
                #Standard Scaling to train and val
                sc = StandardScaler()
                sc.fit(X_train)
                X_train_f = sc.transform(X_train)
                X_test_f = sc.transform(X_test)
                
                #SVC construction
                clf = SVC(kernel='rbf', gamma='scale')
                clf.fit(X_train_f, y_train)
                predtrain = clf.predict(X_train_f)
                predtest = clf.predict(X_test_f)
                
            else:
                
                #SVC construction
                clf = SVC(kernel='rbf', gamma='scale')
                clf.fit(X_train, y_train)
                predtrain = clf.predict(X_train)
                predtest = clf.predict(X_test)               
            
            acc_train = accuracy_score(predtrain,y_train)
            prec_train = precision_score(predtrain,y_train)
            rec_train = recall_score(predtrain,y_train)
            acc_test = accuracy_score(predtest,y_test)
            prec_test = precision_score(predtest,y_test)
            rec_test = recall_score(predtest,y_test)
            
            #Add predictions to the tmp stock preds dict and to the timerange stock preds dict
            tmp_stock_preds_dict['trainpred'+ticker] = predtrain
            tmp_stock_preds_dict['testpred'+ticker] = predtest
            
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'trainpred'+ticker] = predtrain
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testpred'+ticker] = predtest
            
            #Add accuracies precisions and recalls to the tmp stock preds dict and to the timerange stock preds dict
            tmp_stock_preds_dict['trainacc'+ticker] = acc_train
            tmp_stock_preds_dict['trainprec'+ticker] = prec_train
            tmp_stock_preds_dict['trainrec'+ticker] = rec_train
            tmp_stock_preds_dict['testacc'+ticker] = acc_test
            tmp_stock_preds_dict['testprec'+ticker] = prec_test
            tmp_stock_preds_dict['testrec'+ticker] = rec_test
        
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'trainacc'+ticker] = acc_train
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'trainprec'+ticker] = prec_train
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'trainrec'+ticker] = rec_train
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testacc'+ticker] = acc_test
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testprec'+ticker] = prec_test
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testrec'+ticker] = rec_test
            
            #Add test indexes and values to the timerange stock preds dict
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testindexes'] = test_indexes
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'ytest'+ticker] = y_test
            
        #Select the best threshold
        date1 = train_indexes[0]
        date2 = train_indexes[-1]
        df_rets_train = selectdates(df_rets, date1, date2)
        
        CAGR = 0.
        best_threshold = 0.
        for threshold in threshold_list:
            computed_list = computeCAGR(df_rets_train, tickersa, tmp_stock_preds_dict, threshold, threshold_dict, tcosts, mode = 'train', naive = False)
            CAGRval = computed_list[2]
            
            if CAGRval>CAGR:
                CAGR = CAGRval
                best_threshold = threshold
        threshold_repartition_dict[best_threshold]+=1
        print('Best threshold for timerange [{},{}] is {}'.format(str(timerange[0]), str(timerange[1]), best_threshold))
        
        #Backtest with the best threshold
        date3 = test_indexes[0]
        date4 = test_indexes[-1]
        df_rets_test = selectdates(df_rets, date3, date4)
        
        computed_list = computeCAGR(df_rets_test, tickersa, tmp_stock_preds_dict, best_threshold, threshold_dict, tcosts, mode = 'test', naive = False)
        trade_df = computed_list[0]
        
        #Add strategy returns to the timerange stock preds dict
        timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'StratRets'] = trade_df['Strategy Returns']
        
        sharpe = computed_list[1]
        CAGR2 = computed_list[2]
        accuracyD = computed_list[3]
        precisionD = computed_list[4]
        recallD = computed_list[5]
        
        #Add accuraciesD precisionsD recallsD Sharpe and CAGR to the timerange stock preds dict
        timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'accD'] = accuracyD
        timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'precD'] = precisionD
        timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'recD'] = recallD
        timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'sharpe'] = sharpe
        timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'CAGR'] = CAGR2
        
        for ticker in tickersa:
            list_stock_acc_train_tmp.append(tmp_stock_preds_dict['trainacc'+ticker])
            list_stock_prec_train_tmp.append(tmp_stock_preds_dict['trainprec'+ticker])
            list_stock_rec_train_tmp.append(tmp_stock_preds_dict['trainrec'+ticker])
            
            list_stock_acc_test_tmp.append(tmp_stock_preds_dict['testacc'+ticker])
            list_stock_prec_test_tmp.append(tmp_stock_preds_dict['testprec'+ticker])
            list_stock_rec_test_tmp.append(tmp_stock_preds_dict['testrec'+ticker])
        
        avg_stock_acc_train_tmp = np.sum(list_stock_acc_train_tmp)/len(tickersa)
        avg_stock_prec_train_tmp = np.sum(list_stock_prec_train_tmp)/len(tickersa)
        avg_stock_rec_train_tmp = np.sum(list_stock_rec_train_tmp)/len(tickersa)
        avg_stock_acc_test_tmp = np.sum(list_stock_acc_test_tmp)/len(tickersa)
        avg_stock_prec_test_tmp = np.sum(list_stock_prec_test_tmp)/len(tickersa)
        avg_stock_rec_test_tmp = np.sum(list_stock_rec_test_tmp)/len(tickersa)
        
        print('')
        print('In train mode for timerange [{},{}]:'.format(str(train_indexes[0]), str(train_indexes[-1])))
        print('Mean Accuracy train is {} %'.format(avg_stock_acc_train_tmp*100))
        print('Mean Precision train is {} %'.format(avg_stock_prec_train_tmp*100))
        print('Mean Recall train is {} %'.format(avg_stock_rec_train_tmp*100))
        print('Mean Accuracy test is {} %'.format(avg_stock_acc_test_tmp*100))
        print('Mean Precision test is {} %'.format(avg_stock_prec_test_tmp*100))
        print('Mean Recall test is {} %'.format(avg_stock_rec_test_tmp*100))
        print('')
        print('')
        print('In test mode for timerange [{},{}]:'.format(str(test_indexes[0]), str(test_indexes[-1])))
        print('')
        print('AccuracyD is {} %'.format(accuracyD*100))
        print('PrecisionD is {} %'.format(precisionD*100))
        print('RecallD is {} %'.format(recallD*100))
        print('')
        print('Sharpe Ratio is {}'.format(sharpe))
        print('CAGR is {}'.format(CAGR2))
    
        print('')
        print('End timerange: {} to {}'.format(str(timerange[0]), str(timerange[1])))
    
    list_stock_acc_test = list()
    list_stock_prec_test = list()
    list_stock_rec_test = list()
    
    avg_stock_acc_test = 0
    avg_stock_prec_test = 0
    avg_stock_rec_test = 0
    
    pred_metrics_stock = dict()
    
    for ticker in tickersa:
        tot_indexes = list()
        tot_test = list()
        tot_predictions = list()
        for timerange in timeranges:
            temp_indexes = timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testindexes']
            temp_test = list(timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'ytest'+ticker])
            temp_predictions = list(timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testpred'+ticker])
            for (index, test, prediction) in zip(temp_indexes, temp_test, temp_predictions):
                if index not in tot_indexes:
                    tot_indexes.append(index)
                    tot_test.append(test)
                    tot_predictions.append(prediction)
        
        acc_score = accuracy_score(tot_test,tot_predictions)
        prec_score = precision_score(tot_test,tot_predictions)
        rec_score = recall_score(tot_test,tot_predictions)
        
        #Feed the prediction metrics dictionary
        pred_metrics_stock['totindexes'+ticker] = tot_indexes
        pred_metrics_stock['totpredictions'+ticker] = tot_predictions
        pred_metrics_stock['tottest'+ticker] = tot_test
        
        pred_metrics_stock['totacc'+ticker] = acc_score
        pred_metrics_stock['totprec'+ticker] = prec_score
        pred_metrics_stock['totrec'+ticker] = rec_score
        
        list_stock_acc_test.append(acc_score)
        list_stock_prec_test.append(prec_score)
        list_stock_rec_test.append(rec_score)
    
    avg_stock_acc_test = np.sum(list_stock_acc_test)/len(tickersa)
    avg_stock_prec_test = np.sum(list_stock_prec_test)/len(tickersa)
    avg_stock_rec_test = np.sum(list_stock_rec_test)/len(tickersa)
    
    #Retrieve strategy total returns for the whole period
    tot_indexes = list()
    tot_returns = list()
    for timerange in timeranges:
        temp_indexes = timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testindexes']
        temp_returns = list(timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'StratRets'])
        for (index, ret) in zip(temp_indexes, temp_returns):
            if index not in tot_indexes:
                tot_indexes.append(index)
                tot_returns.append(ret)
    
    #Feed the prediction metrics dictionary
    pred_metrics_stock['Strategy Returns'] = tot_returns
    
    #Cumulative returns
    pred_metrics_stock['Cumulative Strategy Returns'] = np.cumsum(pred_metrics_stock['Strategy Returns'])
    
    #Sharpe Ratio
    sharpe = np.sqrt(252)*np.mean(pred_metrics_stock['Strategy Returns'])/np.std(pred_metrics_stock['Strategy Returns'])
    pred_metrics_stock['Sharpe Ratio'] = sharpe
    
    #CAGR
    start_val = 1
    end_val = pred_metrics_stock['Cumulative Strategy Returns'][-1]+1
    start_date = tot_indexes[0]
    end_date = tot_indexes[-1]
    trade_df.index = pd.to_datetime(trade_df.index)
    days = (end_date - start_date).days
    CAGR = (float(end_val) / float(start_val)) ** (252.0/days) - 1
    pred_metrics_stock['CAGR'] = CAGR
    
    print('')
    print('Mean Accuracy test for the whole period is {} %'.format(avg_stock_acc_test*100))
    print('Mean Precision test for the whole period is {} %'.format(avg_stock_prec_test*100))
    print('Mean Recall test for the whole period is {} %'.format(avg_stock_rec_test*100))
    print('')
    print('Whole period Sharpe Ratio is {}'.format(sharpe))
    print('Whole period CAGR is {} %'.format(CAGR*100))
    
    print('')
    
    return timerange_stock_preds_dict, threshold_repartition_dict, pred_metrics_stock

total_weeks = 105
test_weeks = 5
timeranges = create_traintest_bundles(df_ta, ref, total_weeks, test_weeks)

timeranges1 = timeranges[30:40]
ratio = 5/105
test_results = traintest(df_ta, df_rets, tickersa, highrise, ratio, timeranges1, tcosts = True, sc=True)      

pred_metrics_stock = test_results[2]

cumrets = pred_metrics_stock['Cumulative Strategy Returns']
sharpe = pred_metrics_stock['Sharpe Ratio']
CAGR = pred_metrics_stock['CAGR']
        
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(cumrets, color='g', label='Strategy Returns')
plt.legend()
plt.show()

def naivetraintest(df_ta, df_rets, tickersa, highrise, ratio, timeranges, tcosts = False, mode = 'test'):
    threshold_list = [0., 0.04, 0.06, 0.09, 0.12, 0.15, 0.2]
    
    #Supporting Threshold Dict creation
    threshold_dict = dict()
    threshold_dict[round(0.0,1)] = 0.
    threshold_dict[round(0.04,2)] = 0.04
    threshold_dict[round(0.05,2)] = 0.05
    threshold_dict[round(0.06,2)] = 0.7*0.06
    threshold_dict[round(0.09,2)] = 0.09
    threshold_dict[round(0.12,2)] = 0.3*0.09
    threshold_dict[round(0.15,2)] = 0.6*0.15
    threshold_dict[round(0.2,2)] = 0.2*0.2
    
    timerange_stock_preds_dict = dict()
    threshold_repartition_dict = dict()
    
    for threshold in threshold_list:
        threshold_repartition_dict[threshold] = 0
    
    for timerange in timeranges:
        print('')
        print('Start timerange {} to {}'.format(str(timerange[0]), str(timerange[1])))
        
        tmp_stock_preds_dict = dict()
        
        avg_stock_acc_train_tmp = 0
        avg_stock_prec_train_tmp = 0
        avg_stock_rec_train_tmp = 0
        avg_stock_acc_test_tmp = 0
        avg_stock_prec_test_tmp = 0
        avg_stock_rec_test_tmp = 0
        
        list_stock_acc_train_tmp = list()
        list_stock_prec_train_tmp = list()
        list_stock_rec_train_tmp = list()
        list_stock_acc_test_tmp = list()
        list_stock_prec_test_tmp = list()
        list_stock_rec_test_tmp = list()
        
        for ticker in tickersa:
            df_ta_f = selectdates(df_ta, timerange[0], timerange[1])
            highrise_f = selectdates(highrise, timerange[0], timerange[1])
            X = df_ta_f.loc[ticker].copy()
            y = highrise_f.loc[ticker].copy()
            
            l = int((1-ratio)*len(X))
            
            X_train, X_test, y_train, y_test = X[:l], X[l:], y[:l], y[l:]
            
            train_indexes = list(y_train.index)
            test_indexes = list(y_test.index)
            
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)    
            
            predtrain = [1]*len(y_train)
            predtest = [1]*len(y_test)              
            
            acc_train = accuracy_score(predtrain,y_train)
            prec_train = precision_score(predtrain,y_train)
            rec_train = recall_score(predtrain,y_train)
            acc_test = accuracy_score(predtest,y_test)
            prec_test = precision_score(predtest,y_test)
            rec_test = recall_score(predtest,y_test)
            
            #Add predictions to the tmp stock preds dict and to the timerange stock preds dict
            tmp_stock_preds_dict['trainpred'+ticker] = predtrain
            tmp_stock_preds_dict['testpred'+ticker] = predtest
            
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'trainpred'+ticker] = predtrain
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testpred'+ticker] = predtest
            
            #Add accuracies precisions and recalls to the tmp stock preds dict and to the timerange stock preds dict
            tmp_stock_preds_dict['trainacc'+ticker] = acc_train
            tmp_stock_preds_dict['trainprec'+ticker] = prec_train
            tmp_stock_preds_dict['trainrec'+ticker] = rec_train
            tmp_stock_preds_dict['testacc'+ticker] = acc_test
            tmp_stock_preds_dict['testprec'+ticker] = prec_test
            tmp_stock_preds_dict['testrec'+ticker] = rec_test
        
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'trainacc'+ticker] = acc_train
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'trainprec'+ticker] = prec_train
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'trainrec'+ticker] = rec_train
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testacc'+ticker] = acc_test
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testprec'+ticker] = prec_test
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testrec'+ticker] = rec_test
            
            #Add test indexes and values to the timerange stock preds dict
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testindexes'] = test_indexes
            timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'ytest'+ticker] = y_test
            
        #Select the best threshold
        date1 = train_indexes[0]
        date2 = train_indexes[-1]
        df_rets_train = selectdates(df_rets, date1, date2)
        
        CAGR = 0.
        best_threshold = 0.
        for threshold in threshold_list:
            computed_list = computeCAGR(df_rets_train, tickersa, tmp_stock_preds_dict, threshold, threshold_dict, tcosts, mode = 'train', naive=True)
            CAGRval = computed_list[2]
            
            if CAGRval>CAGR:
                CAGR = CAGRval
                best_threshold = threshold
        threshold_repartition_dict[best_threshold]+=1
        print('Best threshold for timerange [{},{}] is {}'.format(str(timerange[0]), str(timerange[1]), best_threshold))
        
        #Backtest with the best threshold
        date3 = test_indexes[0]
        date4 = test_indexes[-1]
        df_rets_test = selectdates(df_rets, date3, date4)
        
        computed_list = computeCAGR(df_rets_test, tickersa, tmp_stock_preds_dict, best_threshold, threshold_dict, tcosts, mode = 'test', naive=True)
        trade_df = computed_list[0]
        
        #Add strategy returns to the timerange stock preds dict
        timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'StratRets'] = trade_df['Strategy Returns']
        
        sharpe = computed_list[1]
        CAGR2 = computed_list[2]
        accuracyD = computed_list[3]
        precisionD = computed_list[4]
        recallD = computed_list[5]
        
        #Add accuraciesD precisionsD recallsD Sharpe and CAGR to the timerange stock preds dict
        timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'accD'] = accuracyD
        timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'precD'] = precisionD
        timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'recD'] = recallD
        timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'sharpe'] = sharpe
        timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'CAGR'] = CAGR2
        
        for ticker in tickersa:
            list_stock_acc_train_tmp.append(tmp_stock_preds_dict['trainacc'+ticker])
            list_stock_prec_train_tmp.append(tmp_stock_preds_dict['trainprec'+ticker])
            list_stock_rec_train_tmp.append(tmp_stock_preds_dict['trainrec'+ticker])
            
            list_stock_acc_test_tmp.append(tmp_stock_preds_dict['testacc'+ticker])
            list_stock_prec_test_tmp.append(tmp_stock_preds_dict['testprec'+ticker])
            list_stock_rec_test_tmp.append(tmp_stock_preds_dict['testrec'+ticker])
        
        avg_stock_acc_train_tmp = np.sum(list_stock_acc_train_tmp)/len(tickersa)
        avg_stock_prec_train_tmp = np.sum(list_stock_prec_train_tmp)/len(tickersa)
        avg_stock_rec_train_tmp = np.sum(list_stock_rec_train_tmp)/len(tickersa)
        avg_stock_acc_test_tmp = np.sum(list_stock_acc_test_tmp)/len(tickersa)
        avg_stock_prec_test_tmp = np.sum(list_stock_prec_test_tmp)/len(tickersa)
        avg_stock_rec_test_tmp = np.sum(list_stock_rec_test_tmp)/len(tickersa)
        
        print('')
        print('For timerange [{},{}]:'.format(str(timerange[0]), str(timerange[1])))
        print('Mean Accuracy train is {} %'.format(avg_stock_acc_train_tmp*100))
        print('Mean Precision train is {} %'.format(avg_stock_prec_train_tmp*100))
        print('Mean Recall train is {} %'.format(avg_stock_rec_train_tmp*100))
        print('Mean Accuracy test is {} %'.format(avg_stock_acc_test_tmp*100))
        print('Mean Precision test is {} %'.format(avg_stock_prec_test_tmp*100))
        print('Mean Recall test is {} %'.format(avg_stock_rec_test_tmp*100))
        print('')
        print('')
        print('In mode {} for timerange [{},{}]:'.format(mode,str(timerange[0]), str(timerange[1])))
        print('')
        print('AccuracyD is {} %'.format(accuracyD*100))
        print('PrecisionD is {} %'.format(precisionD*100))
        print('RecallD is {} %'.format(recallD*100))
        print('')
        print('Sharpe Ratio is {}'.format(sharpe))
        print('CAGR is {}'.format(CAGR2))
    
        print('')
        print('End timerange {} to {}'.format(str(timerange[0]), str(timerange[1])))
    
    list_stock_acc_test = list()
    list_stock_prec_test = list()
    list_stock_rec_test = list()
    
    avg_stock_acc_test = 0
    avg_stock_prec_test = 0
    avg_stock_rec_test = 0
    
    pred_metrics_stock = dict()
    
    for ticker in tickersa:
        tot_indexes = list()
        tot_test = list()
        tot_predictions = list()
        for timerange in timeranges:
            temp_indexes = timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testindexes']
            temp_test = list(timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'ytest'+ticker])
            temp_predictions = list(timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testpred'+ticker])
            for (index, test, prediction) in zip(temp_indexes, temp_test, temp_predictions):
                if index not in tot_indexes:
                    tot_indexes.append(index)
                    tot_test.append(test)
                    tot_predictions.append(prediction)
        
        acc_score = accuracy_score(tot_test,tot_predictions)
        prec_score = precision_score(tot_test,tot_predictions)
        rec_score = recall_score(tot_test,tot_predictions)
        
        #Feed the prediction metrics dictionary
        pred_metrics_stock['totindexes'+ticker] = tot_indexes
        pred_metrics_stock['totpredictions'+ticker] = tot_predictions
        pred_metrics_stock['tottest'+ticker] = tot_test
        
        pred_metrics_stock['totacc'+ticker] = acc_score
        pred_metrics_stock['totprec'+ticker] = prec_score
        pred_metrics_stock['totrec'+ticker] = rec_score
        
        list_stock_acc_test.append(acc_score)
        list_stock_prec_test.append(prec_score)
        list_stock_rec_test.append(rec_score)
    
    avg_stock_acc_test = np.sum(list_stock_acc_test)/len(tickersa)
    avg_stock_prec_test = np.sum(list_stock_prec_test)/len(tickersa)
    avg_stock_rec_test = np.sum(list_stock_rec_test)/len(tickersa)
    
    #Retrieve strategy total returns for the whole period
    tot_indexes = list()
    tot_returns = list()
    for timerange in timeranges:
        temp_indexes = timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'testindexes']
        temp_returns = list(timerange_stock_preds_dict[str(timerange[0])+':'+str(timerange[1])+'StratRets'])
        for (index, ret) in zip(temp_indexes, temp_returns):
            if index not in tot_indexes:
                tot_indexes.append(index)
                tot_returns.append(ret)
    
    #Feed the prediction metrics dictionary
    pred_metrics_stock['Strategy Returns'] = tot_returns
    
    #Cumulative returns
    pred_metrics_stock['Cumulative Strategy Returns'] = np.cumsum(pred_metrics_stock['Strategy Returns'])
    
    #Sharpe Ratio
    sharpe = np.sqrt(252)*np.mean(pred_metrics_stock['Strategy Returns'])/np.std(pred_metrics_stock['Strategy Returns'])
    pred_metrics_stock['Sharpe Ratio'] = sharpe
    
    #CAGR
    start_val = 1
    end_val = pred_metrics_stock['Cumulative Strategy Returns'][-1]+1
    start_date = tot_indexes[0]
    end_date = tot_indexes[-1]
    trade_df.index = pd.to_datetime(trade_df.index)
    days = (end_date - start_date).days
    CAGR = (float(end_val) / float(start_val)) ** (252.0/days) - 1
    pred_metrics_stock['CAGR'] = CAGR
    
    print('')
    print('Mean Accuracy test for the whole period is {} %'.format(avg_stock_acc_test*100))
    print('Mean Precision test for the whole period is {} %'.format(avg_stock_prec_test*100))
    print('Mean Recall test for the whole period is {} %'.format(avg_stock_rec_test*100))
    print('')
    print('Whole period Sharpe Ratio is {}'.format(sharpe))
    print('Whole period CAGR is {} %'.format(CAGR*100))
    
    print('')
    
    return timerange_stock_preds_dict, threshold_repartition_dict, pred_metrics_stock

total_weeks = 40
test_weeks = 5
timeranges = create_traintest_bundles(df_ta, ref, total_weeks, test_weeks)

ratio = 5/40
test_results = naivetraintest(df_ta, df_rets, tickersa, highrise, ratio, timeranges, tcosts = True, mode = 'test')      

pred_metrics_stock = test_results[2]

cumrets = pred_metrics_stock['Cumulative Strategy Returns']
sharpe = pred_metrics_stock['Sharpe Ratio']
CAGR = pred_metrics_stock['CAGR']
        
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(cumrets, color='g', label='Strategy Returns')
plt.legend()
plt.show()
