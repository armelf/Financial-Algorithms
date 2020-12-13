# Trading strategies on Equity & Forex : Proposal of several realistic & optimizable strategies

[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![GitHub license](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://github.com/armelf/Financial-Algorithms/blob/main/LICENSE) (https://github.com/armelf/Financial-Algorithms/graphs/commit-activity) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

This repository proposes a bunch of profitable trading algorithms and trading ideas, designed to be **extensible** and **optimizable**. We focus more on Equity Market for the moment, but with time we will add more asset classes. There is only one algorithm for Forex Market, as we currently don't have enough experience on this market. I hope that the overall project would be of interest and that people will eventually participate and share ideas and knowledge as well, or at least **make improvements** to the strategies already proposed

## Contents

- [Overview](#overview)
- [Preliminaries](#preliminaries)
- [Technical Indicators](#technical-indicators)
  - [Technical Indicators Library](#technical-indicators-library)
  - [Historical price data](#historical-price-data)
  - [Preprocessing historical price data](#preprocessing-historical-price-data)
  - [Usable Features](#usable-features)
  - [Different Strategies and Stock Prediction](#different-strategies-and-stock-prediction)
  - [Creating the dataset](#creating-the-training-dataset)
  - [Backtesting the VWSMA strategy](#backtesting-the-vwsma-strategy)
- [Fundamental trading](#fundamenta-trading)
  - [Data acquisition](#data-acquisition)
  - [Data preprocessing](#data-preprocessing)
  - [Machine learning](#machine-learning)
- [Contributing](#contributing)


## Overview

In general, to make predictions, we proceed as follows:

1. Acquire historical price data – help construct the target variable(what we want to predict) and technical indicators in general.
1. Acquire historical alternative data – It could be fundamental data, news data and so on. They serve as *features* or *predictors*
3. Preprocess data
4. Use a appropriate model or strategy to learn from the data
5. Make predictions with the strategy
6. Backtest the performance of the strategy

## Preliminaries

We use Python 3.7, but over time we will integrate algorithms using other programming languages, depending of the requirements. We mainly retrieve data on Yahoo Finance thans tothe libraries `pandas_datareader` and `yfinance`. The common data science libraries are `pandas` and `scikit-learn`and the library used for Neural Networks treatment is `tensorflow`. We advise you to use Python 3.7 as well, to run .py strategies.  A full list of requirements is included in the `python-requirements.txt` file. To install all of the requirements at once, run the following code in terminal:

```bash
pip install -r python-requirements.txt
```

To get started, clone this project and unzip it.

## Technical Indicators

Strategies you will find here are based on technical analysis. **Technical analysis** is a set of means of examining and predicting price movements in the financial markets, by using historical price charts and market statistics. Thus, those who believe in it suppose that historical data contains relevan information to predict the future, or that historical patterns wil repeat. We test several technical indicators to predict the SPY, which is the S&P500 ETF. Our technical indicators strategies are located in the folder https://github.com/armelf/Financial-Algorithms/tree/main/Equity/Technical%20Indicators

### Technical Indicators Library

We can implement technical indicators by ourselves on Python, since their formulas and the theory behind them are well spread in the finance world. But there already exists Python libraries that implement a lot of Technical Indicators, and that we will use here for our task. The library we are going to work with is `ta`. You will find more information about it here: https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html and the Github page is: https://github.com/bukosabino/ta. To install it run the following code in the terminal: 

```bash
pip install ta
```

### Historical price data

As said above, we retrieve historical prices data from Yahoo Finance and we use `pandas_datareader` to access them easily. This library is very useful since it loads the data from Yahoo directly into `pandas`, in a format that makes them easy to preprocess. They are time-indexed, on a **daily** basis.  A simple syntax to retrieve data is:

```python
from pandas_datareader import data as pdr

ticker = "SPY"
start_date = "2000-01-01"
end_date = "2020-01-01"

#Retrieve yahoo data
data = pdr.get_data_yahoo(ticker, start_date, end_date)
```
### Preprocessing historical price data

Preprocessing historical price data consists of cleaning the price dataset and transforming it to a dataset usable to train and test our strategies. Prices data are retrieved in the OHLC format, plus 2 other columns: **Adj. Close**, the close price adusted to dividends and stock splits, and **Volume**. Data are **time-indexed**. We drop NaN rows from our dataset thanks to the *dropna()* function and we fill empty elements with the function *interpolate()*.  

In our strategies, we try to predict how the **Close Price of the SPY will vary the next day**. For this purpose, we create a target variable named *Price Rise* that is labelled to **1** if the close price **grows tomorrow** and **-1** else.

The code is in the function *create_df* of https://github.com/armelf/Financial-Algorithms/blob/main/Equity/Technical%20Indicators/VWMA-SMA-MeanReversion.py.


### Usable Features

Several technical indicators are used in our strategies:

#### Trend technical indicators

* Simple Moving Average (SMA)
* Exponential Moving Average (EMA)
* Volume Weighted Moving Average (VWMA)
* Moving Average Convergence Divergence (MACD)
* Average Directional Movement Index (ADX)
* Commodity Channel Index (CCI)
* Parabolic Stop And Reverse (Parabolic SAR)

#### Momentum technical indicators

* Relative Strength Index (RSI)
* Stochastic Oscillator (SR)
* Williams %R (WR)

#### Volatility

* Average True Range (ATR)
* Bollinger Bands (BB)

#### Volume

* On-Balance Volume (OBV)

Along with these indicators, we use a special one created to detect market trends, and that will serve in (almost) every single strategy we implement afterwards. The indicator is named *Trend* and is created in the function in the function *create_df* of https://github.com/armelf/Financial-Algorithms/blob/main/Equity/Technical%20Indicators/VWMA-SMA-MeanReversion.py. We compute the percentage change *pc* of the 150-days SMA of the close price and his 150-days standard deviation *sc*. 

- If *pc* > *sc*, we are in an uptrend tomorrow, *Trend* is labelled as 'Uptrend' 
- Elif *pc* < -*sc*, we are in a downtrend tomorrow, *Trend* is labelled as 'Downtrend' 
- Else, there is no trend tomorrow, *Trend* is labelled as 'Range' 

### Different Strategies and Stock Prediction
All technical analysis strategies created are located in https://github.com/armelf/Financial-Algorithms/blob/main/Equity/Technical%20Indicators/technicalindicators_strategies.py

These strategies are characterized by a **signal** that equals **1** if the strategy predicts that tomorrow the market is going up, **-1** if the market is predicted to be going down, and **0** else.

#### Moving Averages Crossover Strategy
We compute the 20-days SMA (20-SMA) and the 50-days SMA (50-SMA) of the close price.

- If *Trend* = 'Uptrend', while 20-SMA >= 50-SMA, signal = 1
- Elif *Trend* = 'Downtrend', while 20-SMA <= 50-SMA, signal = -1
- Else, signal = 0

#### SAR Stochastic Strategy
We compute Parabolic SAR (PSAR), %K and %D of the Stochastic Oscillator. You can find documentation here: 
- Parabolic SAR: https://www.investopedia.com/trading/introduction-to-parabolic-sar/
- Stochastic Oscillator: https://www.investopedia.com/terms/s/stochasticoscillator.asp

We then compute *KminusD* = %K - %D and define a dynamic *stop-loss* (slsar), which is the maximum loss we can bear on one transaction. At every moment **slsar = (SAR-Close)/Close**

We will name the *current order return*, the percentage of return of an outstanding transaction, **cor**

- If *Trend* = 'Uptrend' and SAR < Close and (%K >= 20 and previous%K < 20), while cor>slsar and SAR < Close, signal = 1
- Elif *Trend* = 'Downtrend' and SAR > Close and (%K <= 80 and previous%K > 80), while cor>slsar and SAR > Close, signal = -1
- Else, signal = 0

#### Stochastic MACD Strategy
Again, we compute %K and %D and then *KminusD* = K - %D. We also compute MACD and its signal line(**MACDsl**), and then the MACD signal(**MACDs**) = MACD - MACDsl. You can find documentation here: https://www.investopedia.com/terms/m/macd.asp.

We define a static stop-loss named **sl**.

- If KminusD>0 and MACDs>0, while cor>sl and MACDs>0, signal = 1
- Elif KminusD<0 and MACDs<0, while cor>sl and MACDs<0, signal = -1
- Else, signal = 0

#### RSI Strategy
We compute the 14-days RSI. You can find documentation here: https://www.investopedia.com/terms/r/rsi.asp.

We define a static stop-loss named sl.

- If *Trend* = 'Range' and RSI < 30, while cor>sl and RSI<70, signal = 1
- Elif *Trend* = 'Range' and RSI > 70, while cor>sl and RSI>30, signal = -1
- Else, signal = 0

#### Bollinger Bands RSI Strategy
We compute RSI and Bollinger High and Low Bands(**BBHigh** and **BBLow**). You can find documentation for Bollinger Bands here: https://www.investopedia.com/terms/b/bollingerbands.asp.

We then calculate %B = (Close - BBLow) / (BBHigh - BBLow)

We define a static stop-loss named sl.

- If %B > 0 and previous%B < 0, while cor>sl and %B < 1, signal = 1. This is the *bullish reversal* signal.
- Elif %B < 0.2 and RSI <= 50, while cor>sl and %B < 0.8, signal = 1. Is it a simple Buy signal.
- Elif %B > 0.8 and RSI >= 50, while cor>sl and %B > 0, signal = -1. It is a simple Sell signal.
- Else, signal = 0

#### OBV Bollinger Bands RSI Strategy
We once more compute RSI and Bollinger High and Low Bands, and then %B. After that we compute OBV(On-Balance Volume). You can find documentation about this indicator here: https://www.investopedia.com/terms/o/onbalancevolume.asp.

Finally we calculate the 4-days SMA of RSI, that we call *MRSI*.
We define a static stop-loss named sl.

- If *Trend* = 'Uptrend' and %B > 0.5 and RSI > 50 and (MRSI > previousMRSI and previousMRSI > beforepreviousMRSI) and ((OBV - previousOBV)/previousOBV)>0.005, while while cor>sl and %B < 1, signal = 1.
- Elif *Trend* = 'Downtrend' and %B < 0.5 and RSI <= 50 and (MRSI < previousMRSI and previousMRSI < beforepreviousMRSI) and ((OBV - previousOBV)/previousOBV)<-0.005, while while cor>sl and %B > 0, signal = -1.
- Else, signal = 0.

#### ADX Strategy
We compute the 14-days ADX. You can find documentation here: https://www.investopedia.com/terms/a/adx.asp.

- If *Trend* = 'Uptrend' and ADX > 25, while cor>sl and ADX > 20, signal = 1.
- Elif *Trend* = 'Downtrend' and ADX > 25, while cor>sl and ADX > 25, signal = -1.
- Else, signal = 0

#### CCI ADX Strategy
This strategy is a **reversal strategy**. We compute the 14-days ADX, along with the 20-days CCI. You can find documentation for CCI here: https://www.investopedia.com/terms/c/commoditychannelindex.asp.

- If ADX < 25 and *Trend* = 'Downtrend' and CCI < 100 and previousCCI > 100, while cor>sl and CCI > -100, signal = 1.
- Elif *Trend* = 'Uptrend' and ADX > 25, while cor>sl and ADX > 25, signal = -1.
- Else, signal = 0
