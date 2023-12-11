# json is short for Javascipt Object Notation
#used for data exchange, used for web applications
#know how to encode and decode exchanged data

#Translations
#python: Dict = JSON Object
#python: List, tuples = JSON Array
#python: STR = JSON String
#python: ing, long float = JSON Number
#python: True = JSON true
#python: False = JSON false
#python: Nona = JSON null

import requests
import os
from requests_oauthlib import OAuth1

# Twitter API credentials
API_KEY = 'your_api_key'
API_SECRET_KEY = 'your_api_secret_key'
ACCESS_TOKEN = 'your_access_token'
ACCESS_TOKEN_SECRET = 'your_access_token_secret'

# Setting up the authentication
auth = OAuth1(API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# Endpoint URL for searching tweets
url = "https://api.twitter.com/1.1/search/tweets.json"

# Parameters for the API call
params = {
    'q': 'Python',  # Keyword to search for
    'count': 10,    # Number of tweets to retrieve
    'lang': 'en',   # Language
    'result_type': 'recent'  # Fetch recent tweets
}

# Making the API call
response = requests.get(url, auth=auth, params=params)

# Checking if the request was successful
if response.status_code == 200:
    tweets = response.json()['statuses']
    for tweet in tweets:
        print(f"Tweet: {tweet['text']}\nBy: {tweet['user']['name']}\n")
else:
    print("Failed to fetch tweets")

This script will authenticate with the Twitter API using your credentials and then make a request to the search endpoint to find recent tweets containing the word "Python". It prints out the text of each tweet and the name of the user who posted it.

Remember to replace 'your_api_key', 'your_api_secret_key', 'your_access_token', and 'your_access_token_secret' with your actual Twitter API credentials. Also, handle the data responsibly and in compliance with Twitter's terms of use and privacy policies.