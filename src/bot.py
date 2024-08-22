from dotenv import load_dotenv
import os
import requests
import csv
import json
import re
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

import finnhub

import nltk
nltk.download() # Download NLTK data
from nltk.corpus import stopwords


class Bot():
	"""
	Represents the trading bot

	ATTRIBUTES:
		finnhub_client - The finnhub API client object for this session
		newsapi_key - The API key for newsAPI
		watchlist - An array of 2-value tuples each in form (name, symbol). These are the stocks we will be trading. Loaded from watchlist.csv
		sentiment_forest - A sklearn RandomForestClassifier: The trained random forest for sentiment analysis
	"""


	def __init__(self):

		# Load API keys from .env
		load_dotenv()

		finnhub_key = os.getenv('FINNHUB_KEY')
		self.finnhub_client = finnhub.client(api_key=finnhub_key)

		self.newsapi_key = os.getenv('NEWSAPI_KEY')

		# Load watchlist from watchlist.csv
		with open('../watchlist.csv', newline='') as watchlistCsv:
			reader = csv.reader(watchlistCsv, delimiter=' ')
			self.watchlist = [row for row in reader]

		# Create random forest
		forest = RandomForestClassifier(n_estimators=100)


	def train_forest():
		"""
		Trains the random forest on data from a given CSV file

		PARAMS:
			csv - Name of csv file housing training data
		"""
		self.forest.fit()


	def get_headlines(self, days, company):
		"""
		Gets current date's news headlines from newsAPI that are pertinent to a given company.
		Returned in the form of an array of strings, each string representing the headline of an article

		PARAMS:
			days - Number of days prior to get news headlines from (e.g. days=3 will get headlines from the last 3 days)
			company - Name of the company to get news headlines for
		"""

		# Get a dictionary object of the news (newsAPI returns a JSON object)
		date = datetime.today().strftime('%Y-%m-%d')
		newsJSON = requests.get(f"https://newsapi.org/v2/everything?q={company}&from={date}&sortBy=popularity&apiKey={self.newsapi_key}").json()
		news = json.loads(newsJSON)

		headlines = [ (article['title'] + ". " + article['description']) for article in news['articles'] ] # NewsAPI data seperates headlines into title and description - concatenate these 2 together for our purposes
		return headlines


	def clean_headline(self, headline)
		"""
		Given a headline, "clean" it to prepare for sentiment analysis by:
			- Removing non-letter chars
			- Tokenizing
			- Setting to lowercase
			- Removing stopwords

		returns a cleaned version of the original text, as a string

		PARAMS:
			headline - the headline as a string
		"""

		letters_only = re.sub("[^a-zA-Z]", " ", text) # Remove non-letters 
		tokens = letters_only.lower().split() # Lowercase & tokenize

		# Remove stopwords
		stopword_set = set(stopwords.words('english')) # Convert to set for constant lookup time
		meaningful_word_tokens = [tok for tok in tokens if not tok in stopword_set]

		return ( " ".join(meaningful_word_tokens) ) # Join tokens back together into single string

 
	def est_sentiment(self, company):
		"""
		Returns the estimated sentiment for a given company, based on news found by self.get_headlines()
		The resulting sentiment is based on the headlines of relevant news articles

		PARAMS:
			company - Name of the company
		""" 

		# Get headlines & clean them
		headlines = self.get_headlines(3, company)
		clean_headlines = []
		for headline in headlines:
			cleaned_headline = self.clean_article(headline)
			clean_headlines.append(cleaned_headline)

		# Feature extraction - create vector space for headlines
		vectorizer = CountVectorizer()
		features = vectorizer.fit_transform(clean_headlines)

		result = forest.predict(features)


		