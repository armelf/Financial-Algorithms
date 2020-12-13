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

In our strategies, we try to predict how the **Close Price will vary the next day**. For this purpose, we create a target variable named *Price Rise* that is labelled to **1** if the close price **grows tomorrow** and **-1** else.

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

Along with these indicators, we use a special one created to detect market trends, and that will serve in every single strategy we implement afterwards. The indicator is named *NTrend* and is created in the function in the function *create_df* of https://github.com/armelf/Financial-Algorithms/blob/main/Equity/Technical%20Indicators/VWMA-SMA-MeanReversion.py. We compute the percentage change *pc* of the 150-days SMA of the close price and his 150-days standard deviation *sc*. 

- If *pc* > *sc*, we are in an uptrend tomorrow
- Elif *pc* < -*sc*, we are in a downtrend tomorrow
- Else, there is no trend tomorrow
