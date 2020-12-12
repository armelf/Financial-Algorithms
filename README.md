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

This project uses python 3.6, and the common data science libraries `pandas` and `scikit-learn`. If you are on python 3.x less than 3.6, you will find some syntax errors wherever f-strings have been used for string formatting. These are fortunately very easy to fix (just rebuild the string using your preferred method), but I do encourage you to upgrade to 3.6 to enjoy the elegance of f-strings. A full list of requirements is included in the `requirements.txt` file. To install all of the requirements at once, run the following code in terminal:

```bash
pip install -r requirements.txt
```

To get started, clone this project and unzip it. This folder will become our working directory, so make sure you `cd` your terminal instance into this directory.

## Technical Indicators
