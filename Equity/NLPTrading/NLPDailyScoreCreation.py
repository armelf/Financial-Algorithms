# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:37:20 2020

@author: -
"""


# Application imports
import pandas as pd
from textblob import TextBlob
import json
import math
import re
import os


#Twitter data preprocessing part
def sanitise_tweet(some_string):
    """
    Removes links and special characters
    from a given string
    """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])| (\w +: /  / \S +)", " ", some_string).split())

def get_tweet_sentiment(tweet):
    sanitised_tweet = sanitise_tweet(tweet)
    analysis = TextBlob(sanitised_tweet)
    return analysis.sentiment.polarity

companies = ['AAPL', 'ABBV','ABT','ACN','ADBE','BTC', 'AIG','ALL','AMGN','AMT','AMZN','AXP','BA','BAC',
             'BIIB','BK','BKNG','BLK','BMY','BRK.B','BRK', 'C','CAT','CHTR','CL','CMCS','COF','COP','COST',
             'CRM','CSCO','CVS','CVX','DD','DHR','DIS','DOW','DUK','EMR','EXC','F','FB','FDX',
             'GD','GE','GILD','GM','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KHC',
             'KMI','KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MRK','MS','MSFT',
             'NEE','NFLX','NKE','NVDA','ORCL','OXY','PEP','PFE','PG','PM','PYPL','QCOM','RTX','SBUX','TSLA',
             'SLB','SO','SPG','T','TGT','TMO','TXN','UNH','UNP','UPS','USB','V','VZ','WBA','WFC','WMT','XOM']

#Read json
file_path_keywords = r'YourPathTojsonkeywordsfolder'
keywords_files = os.listdir(file_path_keywords)

startdate = str(keywords_files[0][5:15])
enddate = str(keywords_files[-1][5:15])

columns = []
for company in companies:
    columns.append(company+'_RT_count')
    columns.append(company+'_FV_count')
    columns.append(company+'_RT_RT_count')
    columns.append(company+'_RT_FV_count')
    columns.append(company+'_dailyscore')
    columns.append(company+'_dailyweight')
    
    
df = pd.DataFrame(columns = columns)
idx = pd.date_range(startdate, enddate)
df = df.reindex(idx, fill_value=0)
df = df.fillna(0)
    
for file in keywords_files:
    print('')
    print(file+' begin')
    data = []
    with open(os.path.join(file_path_keywords,file), 'r') as outfile:
        for line in outfile:    
            data.append(json.loads(line))
    
    data = data[0]
    keys = list(data.keys())
    
    dailyweight = dict()
    dailyscore = dict()
    
    for key in keys:
        subdata = data[key][0][::-1]
        for tweet in subdata:
            tokens = tweet['text'].split()
            extracted_companies = list()
            for token in tokens:
                t = token.strip('$').strip('#')
                if t in companies:
                    extracted_companies.append(t)
            if extracted_companies != []:
                tweet_date = tweet['created_at'][:10]
                tweet_message_score = get_tweet_sentiment(tweet['text'])
                tweet_rt_count = tweet['retweet_count']
                tweet_fav_count = tweet['favorite_count']
                tweet_rt_rt_count = tweet['retweeted_retweet_count']
                tweet_rt_fav_count = tweet['retweeted_favorite_count']
                
                
                for company in extracted_companies:
                        
                    dailyweight[company] = dailyweight.get(company, 0.0)
                    dailyscore[company] = dailyscore.get(company, 0.0)
                    try:
                        new_weight = math.log(tweet_rt_count \
                                              +tweet_fav_count \
                                              +tweet_rt_rt_count \
                                              +tweet_rt_fav_count)
                    except ValueError:
                        new_weight = 0
                    
                    new_score = new_weight * tweet_message_score
                        
                    dailyweight[company]+=new_weight
                    
                    dailyscore[company]+=new_score
                            
                    df.loc[tweet_date, company+'_dailyscore']=dailyscore[company]
                    df.loc[tweet_date, company+'_dailyweight']=dailyweight[company]
                    df.loc[tweet_date, company+'_RT_count'] += tweet_rt_count
                    df.loc[tweet_date, company+'_FV_count'] += tweet_fav_count
                    df.loc[tweet_date, company+'_RT_RT_count'] += tweet_rt_rt_count
                    df.loc[tweet_date, company+'_RT_FV_count'] += tweet_rt_fav_count
    print(file+' end')
    

pwd = r"YourPathTodfNLPDailyscore"
df.to_csv(os.path.join(pwd,"df_NLP_daily.csv"))
