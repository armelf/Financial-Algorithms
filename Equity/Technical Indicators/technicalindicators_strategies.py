# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:46:33 2020

@author: ArmelFabrice
"""

import ta
import pandas as pd

def ma_cross_strategy(df, sl, n1 = 20, n2 = 50): #Daily
    #MA20 and MA50 signals
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    #MA20 and MA50 signals
    ma_signal = [0]*len(df)
    ma20 = list(df['MA20'])
    ma50 = list(df['MA50'])
    close = list(df['Close'])
    
    i = 51
    while i < len(df):   
        if df['Trend'][i]=='Uptrend':
            
            #Uptrend
            if ma20[i]>=ma50[i]:
                ma_signal[i] = 1
                count = i+1
                if count<len(df):
                    cur_close = close[count-1]
                    new_close = close[count]
                    pend_ret = (new_close-cur_close)/cur_close
                    while sl<pend_ret  and ma20[count]>ma50[count] and count<len(df)-1:
                        ma_signal[count] = 1
                        count+=1
                        cur_close = close[count-1]
                        new_close = close[count]
                        x = (new_close-cur_close)/cur_close
                        pend_ret+=x
                    if count==len(df)-1 and ma20[count]>ma50[count] and sl<pend_ret:
                        ma_signal[count] = 1
                i = count
            else:
                i+=1
        elif df['Trend'][i]=='Downtrend':
            
            #Downtrend
            if ma20[i]<=ma50[i]:
                ma_signal[i] = -1
                count = i+1
                if count<len(df):
                    cur_close = close[count-1]
                    new_close = close[count]
                    pend_ret = (-1)*(new_close-cur_close)/cur_close
                    while sl<pend_ret  and ma20[count]<ma50[count] and count<len(df)-1:
                        ma_signal[count] = -1
                        count+=1
                        cur_close = close[count-1]
                        new_close = close[count]
                        x = (-1)*(new_close-cur_close)/cur_close
                        pend_ret+=x
                    if count==len(df)-1 and ma20[count]<ma50[count] and sl<pend_ret:
                        ma_signal[count] = -1
                i = count
            else:
                i+=1
        else:
            i+=1
    
    df['MA_signal'] = ma_signal #OK Up Down 40-mean Sharpe 0.516
    return df

def sar_stoch_strategy(df): #Daily 
    #Parabolic SAR
    df['SAR'] = ta.trend.sar(df,af=0.02, amax=0.2)
    
    #Stochastic
    df['Stoch%K'] = ta.momentum.stoch(pd.Series(df['High']),df['Low'], df['Close'], n=14)
    df['Stoch%D'] = ta.momentum.stoch_signal(pd.Series(df['High']),df['Low'], df['Close'], n=14)
    df['KminusD'] = df['Stoch%K'] -df['Stoch%D']
    
    #SAR + Stochastic signals
    df['SAR'] = ta.trend.sar(df,af=0.02, amax=0.2)
    sar = list(df['SAR'])
    stoch = list(df['Stoch%K'])
    kminusd = list(df['KminusD'])
    ss_signal = [0]*len(df)
    close = list(df['Close'])
    
    i = 51
    
    while i < len(df):
        if df['Trend'][i]=='Uptrend':
            #Buy Signals
            if sar[i]<close[i]:
                if (stoch[i]>=20 and stoch[i-1]<=20) or (kminusd[i]>0 and kminusd[i-1]<0):
                    ss_signal[i]= 1
                    count = i+1
                    if count<len(df):
                        cur_close = close[i]
                        new_close = close[count]
                        slsar = (sar[i]-cur_close)/cur_close
                        pend_ret = (new_close-cur_close)/cur_close
                        while slsar<pend_ret  and sar[count]<close[count] and count<len(df)-1:
                            ss_signal[count] = 1
                            slsar = (sar[count]-cur_close)/cur_close
                            count+=1
                            new_close = close[count]
                            pend_ret = (new_close-cur_close)/cur_close
                        if count==len(df)-1 and sar[count]<close[count] and (sar[count]-cur_close)/cur_close<pend_ret:
                            ss_signal[count] = 1
                    i = count
                else:
                    i+=1
            else:
                i+=1
        
        if df['Trend'][i]=='Downtrend':
            #Sell Signals
            if sar[i]>close[i]:
                if (stoch[i]<=80 and stoch[i-1]>=80) or (kminusd[i]<0 and kminusd[i-1]>0):
                    ss_signal[i]= -1
                    count = i+1
                    if count<len(df):
                        cur_close = close[i]
                        new_close = close[count]
                        slsar = (-1)*(sar[i]-cur_close)/cur_close
                        pend_ret = (-1)*(new_close-cur_close)/cur_close
                        while slsar<pend_ret  and sar[count]>close[count] and count<len(df)-1:
                            ss_signal[count] = -1
                            slsar = (-1)*(sar[count]-cur_close)/cur_close
                            count+=1
                            new_close = close[count]
                            pend_ret = (-1)*(new_close-cur_close)/cur_close
                        if count==len(df)-1 and sar[count]>close[count] and (-1)*((sar[count]-cur_close))/cur_close<pend_ret:
                            ss_signal[count] = -1
                    i = count
                else:
                    i+=1
            else:
                i+=1
        else:
            i+=1
    
    df['SS_signal'] = ss_signal #OK Up Down 20-mean Sharpe 0.508
    return df

def stoch_macd_strategy(df,sl):
    
    #Stochastic
    df['Stoch%K'] = ta.momentum.stoch(pd.Series(df['High']),df['Low'], df['Close'], n=14)
    df['Stoch%D'] = ta.momentum.stoch_signal(pd.Series(df['High']),df['Low'], df['Close'], n=14)
    df['KminusD'] = df['Stoch%K'] -df['Stoch%D']
    
    #MACD
    df['MACD'] = ta.trend.macd(pd.Series(df['Close']),n_fast=12, n_slow=26)
    df['MACD_signal_line'] = ta.trend.macd_signal(pd.Series(df['Close']),n_fast=12, n_slow=26, n_sign=9)
    
    #Stochastic + MACD signals
    
    smacd_signal = [0]*len(df)
    macds = list(df['MACD']-df['MACD_signal_line'])
    kminusd = list(df['KminusD'])
    close = list(df['Close'])
    
    i = 51
    while i <len(df):
    #Buy Signals         
#        if df['Trend'][i]=='Uptrend':
        if kminusd[i]>0:
            if macds[i]>0:
                smacd_signal[i] = 1
                count = i+1
                if count<len(df):
                    cur_close = close[i]
                    new_close = close[count]
                    pend_ret = (new_close-cur_close)/cur_close
                    while sl<pend_ret and macds[count]>0 and count<len(df)-1:
                        smacd_signal[count] = 1
                        count+=1
                        new_close = close[count]
                        pend_ret = (new_close-cur_close)/cur_close
                    if count==len(df)-1 and macds[count]>0 and sl<pend_ret:
                        smacd_signal[count] = 1
                i = count
            else:
                i+=1
#        elif df['Trend'][i]=='Downtrend':
        elif kminusd[i]<0:
            if macds[i]<0:
                smacd_signal[i] = -1
                count = i+1
                if count<len(df):
                    cur_close = close[i]
                    new_close = close[count]
                    pend_ret = (-1)*(new_close-cur_close)/cur_close
                    while sl<pend_ret and macds[count]<0 and count<len(df)-1:
                        smacd_signal[count] = -1
                        count+=1
                        new_close = close[count]
                        pend_ret = (-1)*(new_close-cur_close)/cur_close
                    if count==len(df)-1 and macds[count]<0 and sl<pend_ret:
                        smacd_signal[count] = -1
                i = count
            else:
                i+=1
        else:
            i+=1
    
    df['SMACD_signal'] = smacd_signal #OK Up Down 30-mean Sharpe 0.5
    return df

def rsi_strategy(df, sl):

    #RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], n=14)
    
    #Simple RSI Strategy
    rsi_signal = [0]*len(df)
    
    rsi = list(df['RSI'])
    close = list(df['Close'])
    
    i = 51
    while i <len(df):
        #Buy Signals
        if df['Trend'][i]=='Range':
            if rsi[i]<30:
                rsi_signal[i] = 1
                count = i+1
                if count<len(df):
                    cur_close = close[count-1]
                    new_close = close[count]
                    pend_ret = (new_close-cur_close)/cur_close
                    while sl<pend_ret and rsi[count]<70  and count<len(df)-1:
                        x = (new_close-cur_close)/cur_close
                        rsi_signal[count] = 1
                        count+=1
                        cur_close = close[count-1]
                        new_close = close[count]
                        pend_ret += x
                    if count==len(df)-1 and rsi[count]<70 and sl<pend_ret:
                        rsi_signal[count] = 1
                i = count
            
            #Sell signals
            elif rsi[i]>70:
                temp+=1
                rsi_signal[i] = -1
                count = i+1
                if count<len(df):
                    cur_close = close[count-1]
                    new_close = close[count]
                    pend_ret = (-1)*(new_close-cur_close)/cur_close
                    while sl<pend_ret and rsi[count]>30 and count<len(df)-1:
                        x = (-1)*(new_close-cur_close)/cur_close
                        rsi_signal[count] = -1
                        count+=1
                        cur_close = close[count-1]
                        new_close = close[count]
                        pend_ret += x
                    if count==len(df)-1 and rsi[count]>30 and sl<pend_ret:
                        rsi_signal[count] = -1
                i = count
            
            else:
                i+=1
        else:
            i+=1
    
    df['RSI_signal'] = rsi_signal #OK range 10-mean Sharpe 0.69
    return df

        
def bb_rsi_strategy(df,sl):
        
    #Bollinger Bands 
    df['BBHigh'] = ta.volatility.bollinger_hband(df['Close'], n=20, ndev=2, fillna=False)
    df['BBLow'] = ta.volatility.bollinger_lband(df['Close'], n=20, ndev=2, fillna=False)
    df['%B'] = (df['Close']-df['BBLow'])/((df['BBHigh']-df['BBLow']))
    
    #Momentum indicator
    #RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], n=14)
    
    #Bollinger Bands + RSI signals
    rsi = list(df['RSI'])
    B = list(df['%B'])
    bbrsi_signal = [0]*len(df)
    close = list(df['Close'])
    
    i = 51
    while i < len(df):    
        
        #Up-reversal #Average on 30 days   
            #Up-reversal #Average on 30 days  
        if B[i]>0 and B[i-1]<0:
            bbrsi_signal[i] = 1
            count = i+1
            if count<len(df):
                cur_close = close[count-1]
                new_close = close[count]
                pend_ret = (new_close-cur_close)/cur_close
                while sl<pend_ret and B[count]<1 and count<len(df)-1:
                    bbrsi_signal[count] = 1
                    count+=1
                    cur_close = close[count-1]
                    new_close = close[count]
                    x = (new_close-cur_close)/cur_close
                    pend_ret += x
                if count==len(df)-1 and sl<pend_ret:
                    bbrsi_signal[count] = 1
            i = count
        #Buy Signals #30-period 
        elif B[i]<0.2:
            if rsi[i]<=50:
                bbrsi_signal[i] = 1
                count = i+1
                if count<len(df):
                    cur_close = close[count-1]
                    new_close = close[count]
                    pend_ret = (new_close-cur_close)/cur_close
                    while sl<pend_ret and B[count]<0.8 and count<len(df)-1:
                        bbrsi_signal[count] = 1
                        count+=1
                        cur_close = close[count-1]
                        new_close = close[count]
                        x = (new_close-cur_close)/cur_close
                        pend_ret += x
                    if count==len(df)-1 and sl<pend_ret:
                        bbrsi_signal[count] = 1
                i = count
            else:
                i+=1
        
        #Sell Signals #1-day Avg
        elif B[i]>0.8:
            if rsi[i]>=50:
                bbrsi_signal[i] = -1
                count = i+1
                if count<len(df):
                    cur_close = close[count-1]
                    new_close = close[count]
                    pend_ret = (-1)*(new_close-cur_close)/cur_close
                    while sl<pend_ret and B[count]>0 and count<len(df)-1:
                        bbrsi_signal[count] = -1
                        count+=1
                        cur_close = close[count-1]
                        new_close = close[count]
                        x = (-1)*(new_close-cur_close)/cur_close
                        pend_ret += x
                    if count==len(df)-1 and sl<pend_ret:
                        bbrsi_signal[count] = -1
                i = count
            else:
                i+=1
        else:
            i+=1
    
    df['BBRSI_signal'] = bbrsi_signal #OK Up Down 10-mean Sharpe 0.64
    return df


def rsi_obv_bb_strategy(df,sl): #Daily
    #Momentum indicator
    #RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], n=14)
    df['MRSI'] = df['RSI'].rolling(window=4).mean()
    
    #Volume indicator
    #On Balance Volume
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    
    #Bollinger Bands 
    df['BBHigh'] = ta.volatility.bollinger_hband(df['Close'], n=20, ndev=2, fillna=False)
    df['BBLow'] = ta.volatility.bollinger_lband(df['Close'], n=20, ndev=2, fillna=False)
    df['%B'] = (df['Close']-df['BBLow'])/((df['BBHigh']-df['BBLow']))
    
    #RSI + OBV + BB signals
    rob_signal = [0]*len(df)
    obv = list(df['OBV'])
    rsi = list(df['RSI'])
    mrsi = list(df['MRSI'])
    B = list(df['%B'])
    close = list(df['Close'])
    
    i = 51
    
    while i < len(df):
        #Buy Signals
        if df['Trend'][i]=='Uptrend': 
            if B[i]>0.5:
                if rsi[i]>=50 and mrsi[i]>mrsi[i-1] and mrsi[i-1]>mrsi[i-2]:
                    if (obv[i]-obv[i-1])/obv[i-1]>5e-3:
                        rob_signal[i] = 1
                        count = i+1
                        if count<len(df):
                            cur_close = close[i]
                            new_close = close[count]
                            pend_ret = (new_close-cur_close)/cur_close
                            while sl<pend_ret and B[count]<1 and count<len(df)-1:
                                rob_signal[count] = 1
                                count+=1
                                new_close = close[count]
                                pend_ret = (new_close-cur_close)/cur_close
                            if count==len(df)-1 and B[count]<1 and sl<pend_ret:
                                rob_signal[count] = 1
                        i = count
                    else:
                        i+=1
                else:
                    i+=1
            else:
                i+=1
            
        elif df['Trend'][i]=='Downtrend': 
            if B[i]<0.5:
                if rsi[i]<=50 and mrsi[i]<mrsi[i-1] and mrsi[i-1]<mrsi[i-2]:
                    if (obv[i]-obv[i-1])/obv[i-1]<-5e-3:
                        rob_signal[i] = -1
                        count = i+1
                        if count<len(df):
                            cur_close = close[i]
                            new_close = close[count]
                            pend_ret = (-1)*(new_close-cur_close)/cur_close
                            while sl<pend_ret and B[count]>0 and count<len(df)-1:
                                rob_signal[count] = -1
                                count+=1
                                new_close = close[count]
                                pend_ret = (-1)*(new_close-cur_close)/cur_close
                            if count==len(df)-1 and B[count]>0 and sl<pend_ret:
                                rob_signal[count] = -1
                        i = count
                    else:
                        i+=1
                else:
                    i+=1
            else:
                i+=1
        else:
            i+=1
    
    df['ROB_signal'] =  rob_signal #OK Up Down 20-mean Sharpe 0.56    
    return df

def adx_strategy(df,sl): #Daily
    #Average Directional Movement Index
    df['ADX'] = ta.trend.adx(df['High'],df['Low'], df['Close'], n=14)
    
    #ADX Strategy
    adx_signal = [0]*len(df)
    adx = list(df['ADX'])
    close = list(df['Close'])
    
    i = 51
    while i<len(df):        
        #Buy Signals
        if df['Trend'][i]=='Uptrend':
            if adx[i]>25:
                adx_signal[i] = 1
                count = i+1
                if count<len(df):
                    cur_close = close[i]
                    new_close = close[count]
                    pend_ret = (new_close-cur_close)/cur_close
                    while sl<pend_ret and adx[count]>20 and count<len(df)-1:
                        adx_signal[count] = 1
                        count+=1
                        new_close = close[count]
                        pend_ret = (new_close-cur_close)/cur_close
                    if count==len(df)-1 and adx[count]>20 and sl<pend_ret:
                        adx_signal[count] = 1
                i = count
            else:
                i+=1
        elif df['Trend'][i]=='Downtrend':
            if adx[i]>25:
                adx_signal[i] = -1
                count = i+1
                if count<len(df):
                    cur_close = close[i]
                    new_close = close[count]
                    pend_ret = (-1)*(new_close-cur_close)/cur_close
                    while sl<pend_ret and adx[count]>25 and count<len(df)-1:
                        adx_signal[count] = -1
                        count+=1
                        new_close = close[count]
                        pend_ret = (-1)*(new_close-cur_close)/cur_close
                    if count==len(df)-1 and adx[count]>25 and sl<pend_ret:
                        adx_signal[count] = -1
                i = count
            else:
                i+=1
        else:
            i+=1
    
    df['ADX_signal'] = adx_signal #OK Up Down 10-mean Sharpe 0.366
    return df

def cci_adx_strategy(df, sl): #Daily
    #Commodity Channel Index
    df['CCI'] = ta.trend.cci(df['High'],df['Low'], df['Close'], n=20, c=0.015)
    
    #Average Directional Movement Index
    df['ADX'] = ta.trend.adx(df['High'],df['Low'], df['Close'], n=14)
    
    cdx_signal = [0]*len(df)
    cci = list(df['CCI'])
    adx = list(df['ADX'])
    close = list(df['Close'])
    i = 51
    
    #CCI + ADX strategy
    while i<len(df):
        if adx[i]<25:
            #Buy Signals
            if df['Trend'][i]!='Downtrend':
                if cci[i]<100 and cci[i-1]>100:
                    cdx_signal[i] = 1
                    count = i+1
                    if count<len(df):
                        cur_close = close[i]
                        new_close = close[count]
                        pend_ret = (new_close-cur_close)/cur_close
                        while sl<pend_ret and cci[count]>-100 and count<len(df)-1:
                            cdx_signal[count] = 1
                            count+=1
                            new_close = close[count]
                            pend_ret = (new_close-cur_close)/cur_close
                        if count==len(df)-1 and cci[count]>-100 and sl<pend_ret:
                            cdx_signal[count] = 1
                    i = count
                else:
                    i+=1
                    
            else:
                if cci[i]>-100 and cci[i-1]<-100:
                    cdx_signal[i] = -1
                    count = i+1
                    if count<len(df):
                        cur_close = close[i]
                        new_close = close[count]
                        pend_ret = (-1)*(new_close-cur_close)/cur_close
                        while sl<pend_ret and cci[count]<100 and count<len(df)-1:
                            cdx_signal[count] = -1
                            count+=1
                            new_close = close[count]
                            pend_ret = (-1)*(new_close-cur_close)/cur_close
                        if count==len(df)-1 and cci[count]<100 and sl<pend_ret:
                            cdx_signal[count] = -1
                    i = count
                else:
                    i+=1

        else:
            i+=1
    df['CDX_signal'] = cdx_signal #OK Up Down Range 10-mean Sharpe 0.51
    return df

def wr_stoch_strategy(df, sl):
    #William R Indicator
    df['WR'] = ta.momentum.wr(df['High'],df['Low'], df['Close'], lbp=14)
    
    #Stochastic
    df['Stoch%K'] = ta.momentum.stoch(pd.Series(df['High']),df['Low'], df['Close'], n=14)
    df['Stoch%D'] = ta.momentum.stoch_signal(pd.Series(df['High']),df['Low'], df['Close'], n=14)
    df['KminusD'] = df['Stoch%K'] -df['Stoch%D']
    
    #WR Strategy
    wr_signal = [0]*len(df)
    wr = list(df['WR'])
    mwr = list(df['WR'].pct_change().rolling(window=4).mean())
    close = list(df['Close'])
    
    i = 51
    while i<len(df):        
        #Buy Signals
        if wr[i]>-50 and wr[i-1]<-50 and mwr[i]>0:
            wr_signal[i] = 1
            count = i+1
            if count<len(df):
                cur_close = close[i]
                new_close = close[count]
                pend_ret = (new_close-cur_close)/cur_close
                while sl<pend_ret and wr[count]<-20 and count<len(df)-1:
                    wr_signal[count] = 1
                    count+=1
                    new_close = close[count]
                    pend_ret = (new_close-cur_close)/cur_close
                if count==len(df)-1 and wr[count]<-20 and sl<pend_ret:
                    wr_signal[count] = 1
            i = count
        #Sell signals
        elif wr[i]<-50 and wr[i-1]>-50 and mwr[i]<0:
            wr_signal[i] = -1
            count = i+1
            if count<len(df):
                cur_close = close[i]
                new_close = close[count]
                pend_ret = (-1)*(new_close-cur_close)/cur_close
                while sl<pend_ret and wr[count]>-80 and count<len(df)-1:
                    wr_signal[count] = -1
                    count+=1
                    new_close = close[count]
                    pend_ret = (-1)*(new_close-cur_close)/cur_close
                if count==len(df)-1 and wr[count]>-80 and sl<pend_ret:
                    wr_signal[count] = -1
            i = count
        else:
            i+=1
    
    df['WR_signal'] = wr_signal
    return df

def vwsma_strategy(df,sl):
    # VWAP and SMA 20 days
    df['VWAP'] = ta.volume.volume_weighted_moving_average(df['Close'], df['Volume'], n=20)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    #Close price centered and zscore
    df['Close-VWAP'] = df['Close'] - df['VWAP']
    df['zcore1'] = (df['Close-VWAP']-df['Close-VWAP'].rolling(window=40).mean())/(df['Close-VWAP'].rolling(window=40).std())
#    df['zcore1'][50:200].plot(ylim=(-4,4))
    
    zcore = list(df['zcore1'] )
    
    vwsma_signal = [0]*len(df)
    close = list(df['Close'])
    
    i = 51
    #Mean reversion signal
    while i < len(df):  
        if df['Trend'][i]=='Downtrend':
            #Sell signals
            if zcore[i]>1 and zcore[i-1]<1:
                vwsma_signal[i] = -1
                count = i+1
                if count<len(df):
                    cur_close = close[count-1]
                    new_close = close[count]
                    pend_ret = (-1)*(new_close-cur_close)/cur_close
                    while sl<pend_ret and zcore[count]>0 and count<len(df)-1:
                        vwsma_signal[count] = -1
                        count+=1
                        cur_close = close[count-1]
                        new_close = close[count]
                        x = (-1)*(new_close-cur_close)/cur_close
                        pend_ret+=x
                    if sl<pend_ret and count==len(df)-1 and zcore[count]>0:
                        vwsma_signal[count] = -1
                i = count
            else:
                i+=1

        elif df['Trend'][i]=='Uptrend':
            #Buy signals
            if zcore[i]<-1.5 and zcore[i-1]>-1.5:
                vwsma_signal[i] = 1
                count = i+1
                if count<len(df):
                    cur_close = close[count-1]
                    new_close = close[count]
                    pend_ret = (new_close-cur_close)/cur_close
                    while zcore[count]<1.5 and count<len(df)-1:
                        vwsma_signal[count] = 1
                        count+=1
                        cur_close = close[count-1]
                        new_close = close[count]
                        x = (new_close-cur_close)/cur_close
                        pend_ret+=x
                    if count==len(df)-1 and zcore[count]<1.5:
                        vwsma_signal[count] = 1
                i = count
            else:
                i+=1
        else:
            #Buy signals
            if zcore[i]<-2 and zcore[i-1]>-2:
                vwsma_signal[i] = 1
                count = i+1
                if count<len(df):
                    cur_close = close[count-1]
                    new_close = close[count]
                    pend_ret = (new_close-cur_close)/cur_close
                    while sl<pend_ret and zcore[count]<-1 and count<len(df)-1:
                        vwsma_signal[count] = 1
                        count+=1
                        cur_close = close[count-1]
                        new_close = close[count]
                        x = (new_close-cur_close)/cur_close
                        pend_ret+=x
                    if sl<pend_ret and count==len(df)-1 and zcore[count]<-1:
                        vwsma_signal[count] = 1
                i = count
            #Sell signals
            elif zcore[i]>2 and zcore[i-1]<2:
                vwsma_signal[i] = -1
                count = i+1
                if count<len(df):
                    cur_close = close[count-1]
                    new_close = close[count]
                    pend_ret = (-1)*(new_close-cur_close)/cur_close
                    while sl<pend_ret and zcore[count]>1 and count<len(df)-1:
                        vwsma_signal[count] = -1
                        count+=1
                        cur_close = close[count-1]
                        new_close = close[count]
                        x = (-1)*(new_close-cur_close)/cur_close
                        pend_ret+=x
                    if count==len(df)-1 and sl<pend_ret and zcore[count]>1:
                        vwsma_signal[count] = -1
                i = count
            else:
                i+=1

    df['VWSMA_signal'] = vwsma_signal
    return df



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
