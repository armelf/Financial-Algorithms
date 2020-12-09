#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tweepy
import datetime
import time
import os
import json

# credentials from https://apps.twitter.com/
consumer_key = 'YourConsumerKey'
consumer_secret =  'YourConsumerSecretCode'
access_token = 'YourConsumerAccessToken'
access_token_secret = 'YourConsumerTokenCode'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, timeout=180)

users1 = [
    "25073877",     # Donald Trump
    "14886375",     # StockTwits
    "624413",       # MarketWatch #Recurrent
    "16228398",     # Mark Cuban
    "50393960",     # Bill Gates
    "21323268",     # NYSE
    "184020744",    # Mike Flache
    "19546277",     # YahooFinance #Recurrent
    "business",     # Recurrent
    "TechCrunch",
    "WSJ",
    "Forbes",       # Recurrent
    "FT",
    "TheEconomist", # Recurrent
    "nytimes",
    "Reuters",      # Very recurrent
    "GerberKawasaki",
    "jimcramer",
    "TheStreet",
    "TheStalwart",  #Recurrent
    "TruthGundlach",
    "ReformedBroker",
    "bespokeinvest",
    "stlouisfed",
    "federalreserve",
    "GoldmanSachs",
    "ianbremmer",
    "MorganStanley",
    "AswathDamodaran",
    "muddywatersre",
    ]

users2 = [
    "CNBC",    # Recurrent
    "IBDinvestors",
    "nytimesbusiness",
    "Stephanie_Link",
    "jimcramer",
    "WSJmarkets",
    "BreakoutStocks",
    "Benzinga",
    "TraderTVLive",
    "RANsquawk",
    "The_Real_Fly",
    "harmongreg",
    "traderstewie",
    "DanZanger",
    "PeterLBrandt",
    "alphatrends",
    "maoxian",
    "benbernanke",
    "muddywatersre",
    "AswathDamodaran",
    "elerianm",
    "ianbremmer",
    ]


users = users1 + users2

indexes = ["NASDAQ", "Dow", "DJIA", "DJI", "SP500", "FTSE", "CAC40", "STOXX", "DAX", "VIX",
            "NYSE"]

sp100 = ['AAPL', 'ABBV','ABT','ACN','ADBE','BTC', 'AIG','ALL','AMGN','AMT','AMZN','AXP','BA','BAC',
         'BIIB','BK','BKNG','BLK','BMY','BRK.B','BRK', 'C','CAT','CHTR','CL','CMCS','COF','COP','COST',
         'CRM','CSCO','CVS','CVX','DD','DHR','DIS','DOW','DUK','EMR','EXC','F','FB','FDX',
         'GD','GE','GILD','GM','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KHC',
         'KMI','KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MRK','MS','MSFT',
         'NEE','NFLX','NKE','NVDA','ORCL','OXY','PEP','PFE','PG','PM','PYPL','QCOM','RTX','SBUX','TSLA',
         'SLB','SO','SPG','T','TGT','TMO','TXN','UNH','UNP','UPS','USB','V','VZ','WBA','WFC','WMT','XOM']

sp100names = []
keywords = indexes + sp100

#Retrieve data of the penultimate hour on a hourly frequency
#Data come from tweets of users in users and from tweets containing 
#S&P100 companies symbols

while True:
    startDate = datetime.datetime.now()-datetime.timedelta(hours=2)
    endDate =   datetime.datetime.now()-datetime.timedelta(hours=1)
    
    print('Start a new hourly batch')
    print('')
    print('Step 1 started')
    time1 = time.time()
    
    #Tweets by Username
    tweets1 = []
    for username in users:
        tweets1tmp = []
        print('Username {} started'.format(username))
        print('')
        try:
            tmpTweets = api.user_timeline(username)
            for tweet in tmpTweets:
                if tweet.created_at < endDate and tweet.created_at > startDate:
                    dict0 = {}
                    dict0['created_at'] = str(tweet.created_at)
                    dict0['text'] = tweet.text
                    dict0['user_id'] = tweet.user.id_str
                    dict0['Name'] = tweet.user.name
                    dict0['retweet_count'] = tweet.retweet_count
                    dict0['favorite_count'] = tweet.favorite_count
                    dict0['followers_count'] = tweet.user.followers_count
                    try:
                        retfavcount = tweet.retweeted_status.favorite_count
                        retretcount = tweet.retweeted_status.retweet_count
                        dict0['retweeted_retweet_count'] = retretcount
                        dict0['retweeted_favorite_count'] = retfavcount
                    except AttributeError:
                        dict0['retweeted_retweet_count'] = 0
                        dict0['retweeted_favorite_count'] = 0
                    tweets1tmp.append(dict0)
        except tweepy.TweepError as e:
            if 'Failed to send request:' in e.reason:
                print("Time out error caught")
                print('')
                time.sleep(180)
                continue
                
        print('Username {} completed'.format(username))
        print('')
        tweets1+=tweets1tmp

    file_path = r'YourPathTojsonusernamefolder'
    
    try:
        data1 = []
        with open(os.path.join(file_path,'data'+'_'+str(endDate)[:10]+'.json'), 'r') as outfile:
            for line in outfile:    
                data1.append(json.loads(line))
                
        #Increment the id by 1
        data = data1[0]
        identifier = int(list(data.keys())[-1].replace('data'+'_'+str(endDate)[:10]+'_',''))+ 1
        data['data'+'_'+str(endDate)[:10]+'_'+str(identifier)] = tweets1
        with open(os.path.join(file_path,'data'+'_'+str(endDate)[:10]+'.json'), 'w') as outfile:
            json.dump(data, outfile)

        
    except FileNotFoundError:
        data = {}
        identifier = 0
        data['data'+'_'+str(endDate)[:10]+'_'+str(identifier)] = tweets1
        with open(os.path.join(file_path,'data'+'_'+str(endDate)[:10]+'.json'), 'w') as outfile:
            json.dump(data, outfile)
    print('Step 1 completed')
    print('')
    
    print('Step 2 started')
    print('')
    
    #Tweets by Keyword
    tweets2 = []
    counts = []
    for keyword in keywords:
        print('Keyword {} started'.format(keyword))
        tweetsPerQry = 100
        searchQueries = ["#{0}".format(keyword), "${0}".format(keyword)]
        tweets2tmp = []
        
        count = 0
        for query in searchQueries:
            try:
                tmpTweets = api.search(q=query, count=tweetsPerQry)
                for tweet in tmpTweets:
                    if tweet.created_at < endDate and tweet.created_at > startDate:
                        count+=1
                        if tweet.user.followers_count>1:
                            dict0 = {}
                            dict0['created_at'] = str(tweet.created_at)
                            dict0['text'] = tweet.text
                            dict0['user_id'] = tweet.user.id_str
                            dict0['Name'] = tweet.user.name
                            dict0['retweet_count'] = tweet.retweet_count
                            dict0['favorite_count'] = tweet.favorite_count
                            dict0['followers_count'] = tweet.user.followers_count
                            try:
                                retfavcount = tweet.retweeted_status.favorite_count
                                retretcount = tweet.retweeted_status.retweet_count
                                dict0['retweeted_retweet_count'] = retfavcount
                                dict0['retweeted_favorite_count'] = retretcount
                            except AttributeError:
                                dict0['retweeted_retweet_count'] = 0
                                dict0['retweeted_favorite_count'] = 0
                            tweets2tmp.append(dict0)
            except tweepy.TweepError as e:
                if 'Failed to send request:' in e.reason:
                    print("Time out error caught")
                    print('')
                    time.sleep(180)
                    continue
                    
        print('Keyword {} completed'.format(keyword))
        print('')
            
        dictcount= dict()
        dictcount[keyword+str(endDate)[:19]]=count

        tweets2+=tweets2tmp
        counts.append(dictcount)
        
    file_path = r'YourPathTojsonkeywordsfolder'

    try:
        data2 = []
        with open(os.path.join(file_path,'data'+'_'+str(endDate)[:10]+'.json'), 'r') as outfile:
            for line in outfile:    
                data2.append(json.loads(line))
                
        #Increment the id by 1
        data = data2[0]
        identifier2 = int(list(data.keys())[-1].replace('data'+'_'+str(endDate)[:10]+'_',''))+ 1
        data['data'+'_'+str(endDate)[:10]+'_'+str(identifier2)] = (tweets2,counts)
        with open(os.path.join(file_path,'data'+'_'+str(endDate)[:10]+'.json'), 'w') as outfile:
            json.dump(data, outfile)

        
    except FileNotFoundError:
        data = {}
        identifier2 = 0
        data['data'+'_'+str(endDate)[:10]+'_'+str(identifier2)] = (tweets2,counts)
        with open(os.path.join(file_path,'data'+'_'+str(endDate)[:10]+'.json'), 'w') as outfile:
            json.dump(data, outfile)
    print('Step 2 completed')
    print('')
    
    time2 = time.time()
    
    res = time2 - time1
    print('Going to sleep {} seconds'.format(3600-res))
    print('')
    time.sleep(3600-res)
