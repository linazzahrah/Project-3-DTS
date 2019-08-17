# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:00:43 2019

@author: IctUnmul
"""

import tweepy 
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
from googletrans import Translator
translator = Translator()

consumer_key = '3o5QSU17u8EMarZ4jrtjNZiQl'
consumer_secret = 'HkoffeEsxGRWdcBI42Szxk7XLMQ8oz5tXmt2WGNIPNhbH6CJcB'
access_token = '107282554-uVoekWdzR9J9N5YxMypWGf5V6ddsEU8DjtEkmdLT'
access_token_secret = 'c36dMvzMmuLvz9rZjJqxDVXoHJZCYPxdPIJzcP2sPmxgD'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

tweets = api.user_timeline('@SamarindaUpdate', count=200, tweet_mode='extended')
for t in tweets:
    print(t.full_text)
    print()

def list_tweets(user_id, count, prt=False):
    tweets = api.user_timeline(
        "@" + user_id, count=count, tweet_mode='extended')
    tw = []
    for t in tweets:
        tw.append(t.full_text)
        if prt:
            print(t.full_text)
            print()
    return tw

def sentiment_analyzer_scores(text, engl=True):
    if engl:
        trans = text
    else:
        trans = translator.translate(text).text
    score = analyser.polarity_scores(trans)
    lb = score['compound']
    if lb >= 0.05:
        return 1
    elif (lb > -0.05) and (lb < 0.05):
        return 0
    else:
        return -1

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt

def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

def clean_text(text):
    text = re.sub(r"\n", " ", text)
    # remove ascii
    text = _removeNonAscii(text)
    # to lowecase
    text = text.lower()
    return text

def clean_tweets(lst):
    # remove twitter Return handles (RT @xxx:)
    lst = np.vectorize(remove_pattern)(lst, "RT @[\w]*:")
    # remove twitter handles (@xxx)
    lst = np.vectorize(remove_pattern)(lst, "@[\w]*")
    # remove URL links (httpxxx)
    lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")
    # remove punctuations
    lst = np.core.defchararray.replace(lst, "[\w\s]", " ")
    # remove special characters, numbers, punctuations (except for #)
    lst = np.core.defchararray.replace(lst, "[^a-zA-Z#]", " ")
    return lst

def clean_lst(lst):
    lst_new = []
    for r in lst:
        lst_new.append(clean_text(r))
    return lst_new

#def clean_text(text):
#    text = text.lower()
#    text = re.sub(r"what's", "what is ", text)
#    text = text.replace('(ap)', '')
#    text = re.sub(r"\'s", " is ", text)
#    text = re.sub(r"\'ve", " have ", text)
#    text = re.sub(r"this", "this ", text)
#    text = re.sub(r"jln", "jalan", text)
#    text = re.sub(r"aja", "aja ", text)
#    text = re.sub(r"nya", " nya ", text)
#    text = re.sub(r"ya", " ya ", text)
#    text = re.sub(r"deh", " deh ", text)
#    text = re.sub(r"the", "the", text)
#    text = re.sub(r'\s+', ' ', text)
#    text = re.sub(r"\\", "", text)
#    text = re.sub(r"\'", "", text)    
#    text = re.sub(r"\"", "", text)
#    text = re.sub('[^a-zA-Z ?!]+', '', text)
#    text = re.sub('https?://[A-Za-z0-9./]*', '', text)
#    text = _removeNonAscii(text)
#    text = text.strip()
#    return text

def anl_tweets(lst, title='Tweets Sentiment', engl=True ):
    sents = []
    for tw in lst:
        try:
            st = sentiment_analyzer_scores(tw, engl)
            sents.append(st)
        except:
            sents.append(0)
    ax = sns.distplot(sents, kde=False, bins=3)
    ax.set(xlabel='Negative                Neutral                 Positive',
           ylabel='#Tweets',
          title="Tweets of @"+title)
    return sents

stop_words = []
f = open('E:\DTS LINA\project3\stopwords-id.txt', 'r')
for l in f.readlines():
    stop_words.append(l.replace('\n', ''))
f = open('E:\DTS LINA\project3\stopwords.txt', 'r')
for l in f.readlines():
    stop_words.append(l.replace('\n', ''))

additional_stop_words = ['t', 'will']
stop_words += additional_stop_words

def word_cloud(wd_list):
    stopwords = stop_words + list(STOPWORDS)
    all_words = ' '.join([text for text in wd_list])
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        width=1600,
        height=800,
        random_state=21,
        colormap='jet',
        max_words=50,
        max_font_size=200).generate(all_words)
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation="bilinear");

user_id = 'SamarindaUpdate'
count=200
tw_smr = list_tweets(user_id, count)
tw_smr = clean_tweets(tw_smr)
tw_smr[50]
sentiment_analyzer_scores(tw_smr[50])
tw_smr_sent = anl_tweets(tw_smr, user_id)
word_cloud(tw_smr)

dt_smr = {"raw": pd.Series(list_tweets(user_id, count, True))}
tw_smr = pd.DataFrame(dt_smr)
tw_smr['raw'][3]

tw_smr['clean_text'] = clean_lst(tw_smr['raw'])
tw_smr['clean_text'][1]

tw_smr['clean_vector'] = clean_tweets(tw_smr['clean_text'])
tw_smr['clean_vector'][1]

sentiment_analyzer_scores(tw_smr['clean_text'][3],True)

tw_smr['sentiment'] = pd.Series(anl_tweets(tw_smr['clean_vector'], user_id, True))

word_cloud(tw_smr['clean_vector'])
word_cloud(tw_smr['clean_vector'][tw_smr['sentiment'] == 1])
word_cloud(tw_smr['clean_vector'][tw_smr['sentiment'] == -1])