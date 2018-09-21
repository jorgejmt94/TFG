# General:
import tweepy
import re, string

# Twitter App access keys for @user
# Consume:
CONSUMER_KEY    = 'oNaigdsRAxvNLFYkCxTpictTR'
CONSUMER_SECRET = 'I7vv2YhyQXyjAJhi4SDhgrfVLiQbitBVigO03lHA8mQgahIC45'
# Access:
ACCESS_TOKEN  = '474154098-Sr52ZCMC4LVV1A2MnrSD5sQjre8ALPAIBEYjtCOg'
ACCESS_SECRET = '3mb5CJ8M32ijw6F87nyuoZyXoub6lJDdogcR4Nkxuo37l'

# API's setup:
def twitter_setup():
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    # Return API with authentication:
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

def get_twees_by_hashtag(word, n_tweets, year, show):
    api = twitter_setup()
    word='#'+word
    tweets = api.search(q=word, since=str(year)+'-06-14',count=n_tweets)
    if show == True:
        print('Ya tenemos tweets sobre:', word)
    return tweets


def get_last_weets(user_name, n_tweets):
    # create an extractor object:
    extractor = twitter_setup()
    
    # create a tweet list as follows:
    twts = extractor.user_timeline(screen_name=user_name, count=n_tweets)#, tweet_mode='extended')
    tweets=[]
    for tweet in twts:
        #tweet = clean_tweet(tweet)
        tweets.append(get_tweet_data(tweet))
    return tweets

def clean_tweet(tweet):
    #quitar urls
    tweet.text = re.sub(r"http\S+", "", tweet.text).lower()
    #quitar puntos
    tweet.text = tweet.text.replace(".","")
    tweet.text = tweet.text.replace(",", "")
    tweet.text = tweet.text.replace(":", "")
    tweet.text = tweet.text.replace(";", "")
    tweet.text = tweet.text.replace("...", "")
    tweet.text = tweet.text.replace("\"", "")
    tweet.text = tweet.text.replace("#", "")
    return tweet

def clean_tweet_2(tweet):
    #quitar urls
    tweet.text = re.sub(r"http\S+", "", tweet.text).lower()
    #quitar puntos
    tweet.text = tweet.text.replace(".","")
    tweet.text = tweet.text.replace(",", "")
    tweet.text = tweet.text.replace(":", "")
    tweet.text = tweet.text.replace(";", "")
    tweet.text = tweet.text.replace("...", "")
    tweet.text = tweet.text.replace("\"", "")
    tweet.text = tweet.text.replace("#Futbol", "")
    tweet.text = tweet.text.replace("#Baloncesto", "")
    tweet.text = tweet.text.replace("#Golf", "")
    tweet.text = tweet.text.replace("#Boxeo", "")
    tweet.text = tweet.text.replace("#Judo", "")
    tweet.text = tweet.text.replace("#Balonmano", "")
    tweet.text = tweet.text.replace("#Tenis", "")
    tweet.text = tweet.text.replace("#Ciclismo", "")
    tweet.text = tweet.text.replace("#Atletismo", "")
    tweet.text = tweet.text.replace("#Rugby", "")
    tweet.text = tweet.text.replace("#Motociclismo", "")
    tweet.text = tweet.text.replace("#Futbol americano", "")
    tweet.text = tweet.text.replace("#", "")
    return tweet

def get_user_data(user_name, n_tweets):
    import Utils
    #estaria bien saber si es un diario, un periodista, un deportista, un aficionado, mirar si es rt
    # create an extractor object:
    extractor = twitter_setup()
    user = extractor.get_user(user_name)
    return Utils.User(user.description, user.verified,user.followers_count)


def get_tweet_data(tweet):
    import Utils
    return Utils.Tweet(tweet.text, len(tweet.text),tweet.created_at,tweet.source,tweet.favorite_count,tweet.retweet_count, Utils.stem_tokens(tweet.text.split()))


