import config
import praw
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import griddata
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


reddit = praw.Reddit(user_agent= config.my_agent,
                     client_id=config.my_id, client_secret=config.my_secret,
                     username=config.my_username, password=config.my_password)
analyzer = SentimentIntensityAnalyzer()


def main():
    subreddit_name = 'canada'
    sentiment_data = []
    score_data = []
    num_bins = 50
    avg_scores = []
    avg_sentiments = []
    df = pd.DataFrame()
    comments = []

    # fetch comments and get sentiments
    if not os.path.isfile(subreddit_name + '.pkl'):
        json_comments = fetch_comments(subreddit_name)
        for json_comment in json_comments: 
            sentiment = get_sentiment(json_comment.body)
            score = json_comment.score
            comments.append({'sentiment': sentiment, 'score': score})

        #store comments and scores in dataframe
        df = pd.DataFrame(comments)
        df.to_pickle('%s.pkl' % (subreddit_name))
        
    df = pd.read_pickle('%s.pkl' % (subreddit_name))
    #filter out 0 sentiment comments
    df = df[df.sentiment != 0]
    comments = df.to_dict('records')
    
    comments.sort(key=sentiment_sort)
    for comment in comments:
        sentiment_data.append(comment['sentiment'])
        score_data.append(comment['score'])

    # sort data into frequencies and get average score per frequency
    freqs, bins = np.histogram(sentiment_data, num_bins)
    index = 0
    for freq in freqs:
        score_sum = 0
        sentiment_sum = 0
        for offset in range(0, freq):
            score_sum += score_data[index + offset]
            sentiment_sum += sentiment_data[index + offset]
        index += freq
        avg_scores.append(score_sum / freq)
        avg_sentiments.append(sentiment_sum / freq)

    plot_data(sentiment_data, num_bins, avg_scores)
    

def fetch_comments(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)
    top_subreddit = subreddit.top('month', limit=30)
    all_comments = []

    for post in top_subreddit:
        if not post.stickied:
            post.comments.replace_more(limit=None)
            all_comments = all_comments + post.comments.list()
    return all_comments


def get_sentiment(comment):
    sentiment = analyzer.polarity_scores(comment)['compound']
    return sentiment


def sentiment_sort(comment):
    return comment['sentiment']


def plot_data(x, num_bins, z):
    n, bins, patches = plt.hist(x, num_bins)
    plt.xlabel('Sentiment')  
    plt.ylabel('Frequency') 
    plt.title('Comment Sentiment, Frequency, and Score')

    cm = plt.cm.get_cmap('coolwarm')
    col = z - min(z)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
        
    plt.show()

if __name__ == "__main__":
    main()
