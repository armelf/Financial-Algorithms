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
- [Fundamental Trading](#fundamental-trading)
  - [Data acquisition](#data-acquisition)
  - [Data preprocessing](#data-preprocessing)
  - [Machine learning](#machine-learning)
- [NLP Trading](#nlp-trading)
  - [Twitter data retrieval](#twitter-data-retrieval)
  - [Data preprocessing](#data-preprocessing)
    - [Daily scores creation](#daily-scores-creation)
      - [TextBlob Sentiment Analysis](#textblob-sentiment-analysis)
  - [Backtesting](#backtesting)
- [DGuided Strategy](#dguided-strategy)
  - [Data retrieval](#data-retrieval)
  - [Data preprocessing](#data-preprocessing)
  - [D-thresholds & Signal creation](#robustness-&-signal-creation)
  - [Robust Backtesting](#robust-backtesting)
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
- Elif ADX < 25 and *Trend* = 'Uptrend' and CCI > -100 and previousCCI < -100, while cor>sl and CCI < 100, signal = -1.
- Else, signal = 0

#### Williams %R Stochastic Strategy
We compute the %R of Williams(WR). You can find documentation here: https://www.investopedia.com/terms/w/williamsr.asp.

Then, we calculate the 4-days SMA of WR_pct_change = (WR - WRPrevious)/WR_previous and we denote it *mwr*

- If WR > -50 and previousWR < -50 and mwr>0, while cor>sl and WR < -20, signal = 1.
- Elif WR < -50 and previousWR > -50 and mwr<0, while cor>sl and WR > -80, signal = -1.
- Else, signal = 0

#### Volume Weighted Moving Average Strategy
This is a **mean-reversion** strategy. We compute the 20-days Volume Weighted Moving Average(VWMA). You can find documentation here: https://www.investopedia.com/articles/trading/11/trading-with-vwap-mvwap.asp.

This indicator is not implemented by the `ta` library, then you can implement it by yourself is the `volume` module of `ta`. You have the modified `volume` module here: https://github.com/armelf/Financial-Algorithms/blob/main/Equity/Technical%20Indicators/volume.py.

Then, we calculate *CminusVWAP* = Close - VWMA and its rolling zcore, denoted y *zscore* zcore = (CminusVWAP - 40daysSMA(CminusVWAP))/40daysStd(CminusVWAP)

- If *Trend* = 'Downtrend' and zscore > 1 and previouszscore < 1, while cor>sl and zscore > 0, signal = -1.
- *Trend* = 'Uptrend' and zscore < -1.5 and previouszscore > -1.5, while cor>sl and zscore < 1.5, signal = 1.
- Else, if zscore < -2 and previouszscore > -2, while cor>sl and zscore < -1, signal = 1. Elif zscore > 2 and previouszscore < 2, while cor>sl and zscore > 1, signal = -1. Else, signal = 0.

Stock prediction is done this way: **tomorrow_predicted_returns = tomorrows_real_returns * signal**

### Creating the dataset
The final dataset(the one used before backtesting) is created is the *create_df* function of https://github.com/armelf/Financial-Algorithms/blob/main/Equity/Technical%20Indicators/VWMA-SMA-MeanReversion.py. It is composed of OHLCV historical price data, Trend, Strategy Signal(the one used) and Price Rise. In our specific case we will only test the VWSMA strategy, hence the line helps to create the strategy signal:

```python
df = ta_strategies.vwsma_strategy(df, sl)
```

### Backtesting the VWSMA strategy
The backtesting part is done if the function *test_factor_acc* of https://github.com/armelf/Financial-Algorithms/blob/main/Equity/Technical%20Indicators/VWMA-SMA-MeanReversion.py. We create a modified signal(*msignal*) by taking the n-days SMA of the strategy signal. Ou backtestng proves that **n=2** works well for our strategy and it works in a robust way between 1993 and 2019. 

The backtesting is simple, we perform transations one after **one after the other**. Each transaction involves the whole **initial** capital, so we trade **without reinvestment**. Our acktesting is performed **without taking account of transaction costs**, but overall transactions costs are low compared to total strategy returns here, due to the **low turnover** of the strategy (Turnover, in the stock market, refers to the total value of stocks traded during a specific period of time). An **improvement here would be to compute strategy returns taing transaction costs into account**. 

We use *log-returns* = ln(Close/previousClose) for our backtesting and are aware of the *look-ahead bias*, then we use today's modified signal to compute tomorrow returns(*trets*).

Note *srets* our outstanding strategy returns. 
- If msignal = 1, srets = trets
- Elif msignal = -1, srets = -trets
- Else, srets = 0

At the end we compute cumulative returns *cumrets* = 1 + sum(srets) and plot the *equity_curve*.

Between 1993 and 2019 we obtain:
![VWSMA Strategy Graph](Equity/Technical%20Indicators/VWSMA%20Strategy%20Returns.png)

## Fundamental Trading

We aim to predict companies share prices thanks to their fundamental valuations. We collect fundamental data thanks to the Python library named `simfin` and we use them to compute needed valuations for our predictions. We then apply Machine Learning techniques to make our predictions.

The strategy we are going to show is mainly inspired from Robert Martin's project that you will find here: https://github.com/robertmartin8/MachineLearningStocks

### Data acquisition

SimFin is a platform where we can retrieve free fundamental data until one year efore the retrieval date. `simfin` is the corresponding Python API. You will find more information about it here: https://simfin.com/ and the Github page is: https://github.com/SimFin/simfin-tutorials. To install it run the following code in the terminal: 

```bash
pip install simfin
```

Sinfin proposes fundamental data for **a lot of company**, of a *quarterly*, *annual* and *ttm (Twelve Trailing Months)* basis. The platform also gives access to historical prices data on a **daily basis**. We will use this abondance of data to develop a **long-only multi-stocks strategy** and at each potential trading time, we only pick and trade on a certain number of stocks.

Below are the interesting fundamental metrics, available on SimFin, that will help us to construct our set of features:

#### Income Statement metrics

- Revenue
- Cost of Revenue
- Shares(Diluted)
- Gross Profit
- Net Income
- Net Income Available to Common Shareholders 
- Operating Income or Loss

#### Balance Sheet metrics

- Long-term debt
- Shares(Diluted)
- Total Equity
- Cash, Cash Equivalents & Short Term Investments (Total Cash)
- Total Assets
- Total Current Assets
- Total Current Liabilities

#### Cash-flows Statement metrics

 - Net Cash from Operating Activities (Operating Cash Flow)
 - Change in Fixed Assets & Intangibles
 
#### Historical prices dataset
 
 - OHLC
 - Adjusted Close
 - Shares Outstanding

A simple syntax to retrieve data is:

```python
import simfin as sf
sf.set_api_key('free')

# Set the local directory where data-files are stored.
# The dir will be created if it does not already exist.
sf.set_data_dir('YourDataDirPath')

market = 'us'

# TTM Income Statements.
df_income_ttm = sf.load_income(variant='ttm', market=market)

# Quarterly Income Statements.
df_income_qrt = sf.load_income(variant='quarterly', market=market)

# TTM Balance Sheets.
df_balance_ttm = sf.load_balance(variant='ttm', market=market)

# Daily Share-Prices.
df_prices = sf.load_shareprices(variant='daily', market=market)
```
Note that every *load* function of `simfin` will download data(for every single stock available) from the online platform, save it into *YourDataDirPath* and then load it on your Python session.

The syntax helps you to save fundamental data for the set of companies tickers you want to use in your strategy:

```python
df_income_ttm = df_income_ttm.loc[tickers].copy()
```

### Data preprocessing
Our preprocessing mainly consists in creating our set of features and our *target variable y* (The variable we want to predict and that will serve to run our strategy). 
First, define the set of variables used: 

#### Valuation fundamentals

- Market Capitalization https://www.investopedia.com/terms/m/marketcapitalization.asp#:~:text=Market%20capitalization%20refers%20to%20the,market%20price%20of%20one%20share
- Enterprise Value https://www.investopedia.com/terms/e/enterprisevalue.asp
- Price-to-Earnings Ratio (P/E) https://www.investopedia.com/terms/p/price-earningsratio.asp
- Price/Earnings-to-Growth Ratio (PEG) https://www.investopedia.com/terms/p/pegratio.asp
- Price/Sales Ratio https://www.investopedia.com/terms/p/price-to-salesratio.asp
- Price/Book Ratio https://www.investopedia.com/terms/p/price-to-bookratio.asp#:~:text=Companies%20use%20the%20price%2Dto,value%20per%20share%20(BVPS).
- Enterprise Value / Revenue
- Enterprise Value / EBITDA

#### Financial fundamentals

- Profit Margin https://www.investopedia.com/terms/p/profitmargin.asp
- Operating Margin https://www.investopedia.com/terms/o/operatingmargin.asp
- Return on Assets https://www.investopedia.com/terms/r/returnonassets.asp
- Return on Equity https://www.investopedia.com/terms/r/returnonequity.asp
- Revenue
- Revenue Per Share
- Quarterly Revenue Growth
- Gross Profit https://www.investopedia.com/terms/g/grossprofit.asp
- EBITDA https://www.investopedia.com/terms/e/ebitda.asp
- Net Income
- Net Income Available to Common Shareholders (Earnings)
- Diluted Earnings-Per-Share https://www.investopedia.com/terms/d/dilutedeps.asp
- Quarterly Earnings Growth
- Total Cash
- Total Cash Per Share
- Total Debt
- Total Debt/Equity
- Current Ratio https://www.investopedia.com/terms/c/currentratio.asp
- Book Value Per Share https://www.investopedia.com/terms/b/bvps.asp
- Operating Cash Flow
- Free Cash Flow https://www.investopedia.com/terms/f/freecashflow.asp

#### Technical indicators

- Beta https://www.investopedia.com/terms/b/beta.asp
- 50-days Close Price SMA
- 200-days Close Price SMA
- 60-days Volume SMA
- Shares Outstanding

After the calculus of all these indicators is done for every stock of our *tickers* list, we compute the *forward semester returns(%)(FSV)* = 100 * (125daysForwardClose - Close) / Close
We also compute the forward semester returns of the SPY: *FSVSPY*
The principle of the strategy is simple: fundamentals help us to predict the **semiannually outperformance of a stock relative to the SPY**. We then define an *outperformance* parameter, that is adjustable. Our dataset is time-indexed on a *quarterly basis*. You can find the code here: https://github.com/armelf/Financial-Algorithms/blob/main/Equity/Fundamental%20Trading/FundDatasetCreation.py

For one stock, at a certain trading date:
- If FSV > FSVSPY + outperformance, y = 1 
- Else, y = 0
 
### Machine learning
Now we have our features and our target variable, we can apply Machine Learning techniques to learn patterns in our data and predict on new samples. We use the classical technique of *train-test split*, knowing our dataframe is time-indexed and sorted by indexed, we don't use data from the future in our training process (train set). We then avoid the *look-ahead bias*. Our test set serves to predict, using our model already trained on and having learned from the train set. It is a classification problem. The set of fundamental + technical features created help to predict y, variable that has 2 classes (labels): 0 and 1. We use a *Random Forest Classifier* and you can see the code here: https://github.com/armelf/Financial-Algorithms/blob/main/Equity/Fundamental%20Trading/FundTradingAlgo.py.
The trained ML algorithm is feeded with *X_test* values where *X_test* is the set of features of the test set, and predicts *y_pred* value, which wille correspond with *y_test* that contains the real target values of the test set. This way, we can have an *accuracy_score*that predicts to evaluate our prediction.

It is known that financial markets behaviour vary with time, due to macroeconomic, political, or just economic changes. Our whole dataset goes from 2009 to 2019. But the way fundamentals affected the market in 2009 is not the same they will affect the maret in 2019. To make our training consistent with our testing, we need our training data to behave a bit like our testing data. We split our whole dataset such that *we train on 3 quarters and we predict on the next one, and we roll over the entire backtesting period*.

Strategy returns for every stock are computed as signal * FSV and for every test period we compare the mean strategy returns of all the stocks traded with the SPY returns for the same period. Results are printed in this .txt file: https://github.com/armelf/Financial-Algorithms/blob/main/Equity/Fundamental%20Trading/FundTradingAlgoResults20092019.txt

Here are some results, in 2015:
```txt
From 2015-06-30 to 2015-07-31

Classifier performance
 ====================
Accuracy score:  0.67
Precision score:  0.67

 Stock prediction performance report 
 ========================================
Total Trades: 58
Average return for stock predictions:  9.0 %
Average market return in the same period: -1.2% 
Compared to the index, our strategy earns  10.2 percentage points more

From 2015-07-31 to 2015-08-31

Classifier performance
 ====================
Accuracy score:  0.74
Precision score:  0.75

 Stock prediction performance report 
 ========================================
Total Trades: 12
Average return for stock predictions:  11.4 %
Average market return in the same period: -3.9% 
Compared to the index, our strategy earns  15.3 percentage points more

From 2015-08-31 to 2015-09-30

Classifier performance
 ====================
Accuracy score:  0.59
Precision score:  0.70

 Stock prediction performance report 
 ========================================
Total Trades: 91
Average return for stock predictions:  13.8 %
Average market return in the same period:  6.5% 
Compared to the index, our strategy earns  7.3 percentage points more
```

## NLP Trading

In this part, we assume that Tweets can help to predict stocks. The rationale is that we can extract sentiment from tweets, that would help to predict future stock movements. Obviously, we retrieve tweets from Twitter. The problem is that we are limited in the amount of tweets we can retrieive from Twitter. To tackle that, we created a Twitter crawler that ran between May & Octoer 2020. You can find the Crawler here: https://github.com/armelf/Financial-Algorithms/blob/main/Equity/NLPTrading/TwitterCrawler.py.

### Twitter data retrieval

As said above, we retrieved data from Twitter, everyday between May and October 2020. We then have a 5-months tweets history. We use the Python library `tweepy` that you can find here: https://www.tweepy.org/. It is a Python library to access the Twitter API. The corresponding Github is: https://github.com/tweepy/tweepy. 

Here is the syntax to install this library:
```bash
pip install tweepy
```

To get access to Twitter data through the API, you need to go on the Twitter Developer website https://developer.twitter.com/apps and to apply for access. You will then be given some credentials:

 - Your Consumer Key
 - Your Consumer Secret Code
 - Your Access Token
 - Your Access Token Secret Code
 
 #### Data crawling
 The principle of how data crawler is simple: every hour, we retrieve several features of the tweets of:
 - People in a list of users, a list named *users*. Our users are made up of well-known people, investors and analysts in the stock markets. For example Bill Gates, analysts from the NYSE, and so on.
 - A keywords list named *keywords*, composed of the 100 components of the S&P100. We want to retrieve tweets relating events about one or several components of the S&P100.
 
 A tweet is retrieved as a *dictionary* characterized by several keys or features you can find in the Tweet Obect Model here: https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/tweet.
 
 Hence, every hour, for every user and for every keyword, we retrieve all the tweets posted over the penultimate hour. We especially retrieved these features or keys of the tweets:
 
 - 'created_at': The date which the tweet message was posted
 - 'text': The tweet message
 - 'user_id': The user identifier
 - 'name': The user name
 - 'retweet_count': The number of retweets at the retrieval time
 - 'favorite_count': The number of favorites at the retrieval time
 - 'retweet_followers_count': The number of followers at the retrieval time
 
 If the tweet was retweeted, we also retrieve:
 - 'retweeted_retweet_count': The number of retweets of the retweeted tweets at the retrieval time
 - 'retweeted_favorite_count': The number of favorites of the retweeted tweets at the retrieval time

Every day we retrieved this data and stored it in daily .json files. Once again, the whole crawling process is here: https://github.com/armelf/Financial-Algorithms/blob/main/Equity/NLPTrading/TwitterCrawler.py.

### Data preprocessing
After having stored Financial Twitter data in daily .json for 5 months, we decided to use them to implement our strategy. We begin by preprocessing our Twitter data in order to create a pandas Dataframe that will contains a set of features, created thanks to our Twitter data, and suitable to make a daily trading strategy. In the preprocessing part of this file: https://github.com/armelf/Financial-Algorithms/blob/main/Equity/NLPTrading/NLPDailyScoreCreation.py, we create a *for loop* that will visit all Twitter data in relationship with *keywords*(the list containing all the tickers of the S&P100). We loop over each daily .json file, and retrieve twitter features cited above to compute, for each company in the S&P100, a daily_score named *daily_score* that will help to create our daily signals(we have one signal for each company / stock). The process is described below:

#### Daily scores creation
For a specific day and a specific company:
- dailyscore = 0
- For each Tweet of the daily json file corresponding to the date: 
  - If the company's ticker is detected in the Tweet message (*text*), retrieve *retweet_count*, *favorite_count*, *retweeted_retweet_count*, *retweeted_favorite_count*. 
  Denote *s* = retweet_count + favorite_count + retweeted_retweet_count + retweeted_favorite_count
    - Create *new_weight* = ln(s) and create *tweet_message_score* thanks to the sentiment analysis of *text*. *new_score* = new_weight * tweet_message_score
    - Update the daily score of the company: dailyscore = dailyscore + new_score
   - Else: Do nothing
   
##### TextBlob Sentiment Analysis
We explain here how we compute our message score. We use the Python library named `textblob`. It is a Python library that provides a simple API for common NLP tasks. Here is a tutorials website: https://textblob.readthedocs.io/en/dev/quickstart.html. You can install it typing this syntax in your terminal:

```bash
pip install textblob
```
`textblob` provides *polarity*(score)(range of [-1,1]) and *subjectivity*(score)(range of [0,1]) estimates for parsed documents, through the *.sentiment* attribute of its *TextBlob* class. The overall sentiment is often inferred as **positive, neutral or negative from the sign of the polarity score**. Subjective sentences or texts generally refer to personal opinion, emotion or judgment whereas objective refers to factual information. The subjectivity score reflects how much subjective is our text. Our tweet_message_score corresponds to the **polarity** of the tweet text.

*NB.* An improvement here could be to juggle between polarity and subjectivity to find a more relevant tweet_message_score

### Backtesting

Now that we have daily NLP scores for every stock in our portfolio (S&P100 stocks in our case), we are able to create a trading signal *s*, that aims to predict stock prices changes for the next day. Note that we are *long-only* in this strategy. Note that **due to the lack of data(only 5 months of daily data), we introduce a look-ahead bias in our strategy** by creating relevant threshold using the whole dataset. 

For each company, denote *pos_mean_scores*, *neg_mean_scores*, *pos_scores_std*, *neg_scores_std* = average (standard deviation) of the positive(negative) daily scores of the company over the whole dataset. They will serve as thresholds for our strategy. *A lookaround would haveeen to use rolling mean scores and rolling standard deviations as well*, ut we don't have enough data to do it.

Every day:

- If daily_score > pos_mean_scores + 2 * pos_scores_std, s = 1. It is a simple Buy Signal
- Elif daily_score < neg_mean_scores - 3 * neg_scores_std, s = 1. It is *Reversal* Buy Signal.

Using for tomorrow returns and outstanding strategy returns the same notation as in our previous strategies, we then create a portfolio where, for each company, srets = trets * s.
Our daily strategy returns, denoted y *avgsrets* = average, for each company having a signal(such that s = 1), of these companies tomorrow returns.

We compute cumulative returns *cumrets* = 1 + sum(avgsrets) and plot the *equity_curve*.

Between May & October 2020, we obtain:
![NLP Strategy Graph](Equity/NLPTrading/NLPTradingReturnsBetweenMay2020%26Oct2020.PNG)

## DGuided Strategy
We are going to illustrate here a trading strategy that deals with high prices changes to predict stock close prices for the following week. We will make predictions on a portfolio composed of the 500 stocks of the S&P500. The whole code is here: https://github.com/armelf/Financial-Algorithms/blob/main/Equity/Robust%20Strategies/ThresholdWeeklyStrategies.py

### Data retrieval
We retrieve historical prices data from `simfin`, with the Python syntax: 

```python
import simfin as sf
sf.set_api_key('free')

# Set the local directory where data-files are stored.
# The dir will be created if it does not already exist.
sf.set_data_dir('YourDataDirPath')

market = 'us'

# Daily Share-Prices.
df_prices = sf.load_shareprices(variant='daily', market=market)
```
Since we will use S&P500 companies, we need to retrieve the corresponding list of tickers, which is accessible here on Wikipedia: http://en.wikipedia.org/wiki/List_of_S%26P_500_companies. We will then use a library named `html_table_parser`, whose purpose is to parse HTML tables without help of external modules . The syntax to install this library is:

```bash
pip install html-table-parser-python3
```
Since we want to make weekly predictions, when data are retrieved, we will **resample our dataset on a weekly basis**.

### Data preprocessing
Now that the dataset has been resampled to weekly basis, we create our features is the function *calc()*. We will eventually use could some technical indicators, so we compute them as well. In this strategy we will not use them, but **it could be a point of improvement** to use and optimize our set of features + Machine Learning to create our signal. In fact the commented function *traintest()* uses technical features along with a Machine Learning Classifier, the Support Vector Machines Classifier *SVC*, and with train-test techniques, creates a signal. You have some information about Support Vector Machines (SVM) here: https://scikit-learn.org/stable/modules/svm.html. The technical indicators computed(each applied on the Close Price) are:

- 10-weeks Simple Moving Average
- 20-weeks Exponential Moving Average
- 10-weeks Zero Lag Exponential Moving Average (https://www.technicalindicators.net/indicators-technical-analysis/182-zlema-zero-lag-exponential-moving-average)
- 10-weeks Weighted Moving Average(https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/wma)
- 14-weeks RSI
- 10-weeks Momentum. It is the percentage of change of the Close price between week t and day t-10.
- 10-weeks Rate Of Change (ROC) (https://www.investopedia.com/terms/r/rateofchange.asp#:~:text=What%20is%20Rate%20of%20Change%20(ROC)&text=ROC%20is%20often%20used%20when,the%20slope%20of%20a%20line.)
- Force Index (https://www.investopedia.com/terms/f/force-index.asp)

We want to remind that we don't use these technical indicators in our strategy, but we can. The core of this strategy is in the function *naivetraintest()* of the .py file. As the name of the function indicates, it is a **Naive** strategy, only using a set of simple and static principles to make our predictions.

We build several other features that will be used to construct our signal:
- Forward Close - Close returns (*FCCrets*). It is a complicated name, for the returns of the next week
- High Close returns (*HCrets*): The relative variation between the High price and the Close price of the current week
- Previous High Close returns (*PHCrets*): The relative variation between the High price and the Close price of the previous week
- Forward High Close returns (*FHCrets*): The relative variation between the High price of the next week and the Close price of the current week
- Forward High High returns (*FHHrets*): The relative variation between the High price of the next week and the High price of the current week. 

HCrets, PHCrets and FHHrets are the **most important** features for the signal creation of our naive strategy.

Once all these features are created, we apply the *dropna()* function to our dataset to remove possible NaN rows.

### D-thresholds & Signal Creation
This strategy is called D-Guided because we use a set of rules around a dictionary of threshold, *threshold_dict*. 'D' corresponds to one occurence of a dictionary (key + value). The keys(*threshold_key*) of the dictionary threshold_dict are composed of static thresholds on HCrets. Their correspond values(*threshold_value*) are *smoothed* static thresholds. This strategy is a **long-only breakout-strategy**.

For each company, at a certain trading time:
- If HCrets > threshold_key and PHCrets > threshold_value, Buy at the Close Price and place a stop-loss at High Price. The consequence is that: 
  - If FHHrets > 0 (a Higher High next week), the outstanding strategy returns srets = HCrets
  - Else, the outstanding strategy returns srets = FCCrets
- Else, do nothing.

Our weekly strategy returns, denoted y *avgsrets* = average, for each company having a srets!=0.0, of these companies srets.

We compute cumulative returns *cumrets* = 1 + sum(avgsrets) and can plot the *equity_curve*.

The code of this signal creation and even (mere) backtesting part is in the function *computeCAGR()*. The whole backtesting process is a little more complicated. This function also proposes D-accuracy, D-precision and D-recall scores, that are accuracy, precision and recall scores of our threshold decision. You will find documentation on these scores here: https://wiki.pathmind.com/accuracy-precision-recall-f1. Finally the function provides a Sharpe Ratio and a CAGR: https://www.investopedia.com/terms/s/sharperatio.asp and https://www.investopedia.com/terms/c/cagr.asp. 

### Robust backtesting
The whole backtesting process is performed in the function *naivetraintest()*, this backtesting is qualified of *robust* because we split our dataset in several relatively little bundles, large enough to learn from our strategy and make predictions, but little enough to tackle changes in the financial market over time. We already used this method in our Fundamental Trading strategy below. But here, for each bundle:

- We perform a 0.75-0.25 train-test split. The train set helps to find the best D-threshold over the training period, in threshold_dictionary. 

- We select the D-threshold that gives the highest CAGR over the training period. Note that when choosing our training D-threshold, **we take transaction costs (tc) into account**. In this strategy, we estimate transaction costs around *0.35% (35 pips)* of our initial transaction amount. Then we backtest over the testing period, using this D-threshold and transaction costs and we roll over the whole data set.

We need to specify that there exists a transaction costs parameter in the *naivetraintest()* and *computeCAGR()* functions, that is labeled 'True' if we consider transaction costs and 'False' else.

Backtesting is done between 2007 and 2019. The .txt file here: https://github.com/armelf/Financial-Algorithms/blob/main/Equity/Robust%20Strategies/NaiveThresholdStrategieResults20072019 transcribes relevant strategy figures during the whole backtesting period. 

Here is an overview: 
```txt
Start timerange 2017-02-27 to 2017-09-15
Best threshold for timerange [2017-02-27,2017-09-15] is 0.12

For timerange [2017-02-27,2017-09-15]:
Mean Accuracy train is 54.236874236874236 %
Mean Precision train is 100.0 %
Mean Recall train is 54.236874236874236 %
Mean Accuracy test is 53.205128205128204 %
Mean Precision test is 99.48717948717949 %
Mean Recall test is 53.205128205128204 %


In mode test for timerange [2017-02-27,2017-09-15]:

AccuracyD is 14.285714285714285 %
PrecisionD is 14.285714285714285 %
RecallD is 100.0 %

Sharpe Ratio is 4.828744845310482
CAGR is 1.7834698489282919

End timerange 2017-02-27 to 2017-09-15
```

Here, you will find the Equity Curve between 2007 & 2019, *transaction costs included* we obtain:
![D-Guided Strategy Graph](Equity/Robust%20Strategies/NaiveThresholdStrategyEquityCurve20072019.PNG)

*NB.* This strategy **needs** optimization and there is room for improvement. Every suggestion is welcome, and a first one could be to train on a wider range of D-thresholds, eventually dynamic(and not static) ones
