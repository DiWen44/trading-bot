from dotenv import load_dotenv
import os
import requests
import csv
import json
import re
from datetime import datetime

import pandas as pd
import numpy as np

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


	def get_news(self, days, company):
		"""
		Gets current date's news articles from newsAPI for a given company.
		Returned in the form of an array of strings, each string representing the content of an article

		PARAMS:
			days - Number of days prior to get news from (e.g. days=3 will get headlines from the last 3 days)
			company - Name of the company to get news for
		"""

		# Get a dictionary object of the news (newsAPI returns a JSON object)
		date = datetime.today().strftime('%Y-%m-%d')
		newsJSON = requests.get(f"https://newsapi.org/v2/everything?q={company}&from={date}&sortBy=popularity&apiKey={self.newsapi_key}").json()
		news = json.loads(newsJSON)

		articles = [ article['content'] for article in news['articles'] ]
		return articles



	def clean_article(self, text)
		"""
		Given the content of an article, "clean" it to prepare for sentiment analysis by:
			- Removing non-letter chars
			- Tokenizing
			- Setting to lowercase
			- Removing stopwords

		returns a tokenized (i.e. array of words) and cleaned version of the original text

		PARAMS:
			Text - the content of the article
		"""

		letters_only = re.sub("[^a-zA-Z]"," ", text) # Remove non-letters 
		tokens = letters_only.lower().split() # Lowercase & tokenize

		# Remove stopwords
		tokens = [tok for tok in tokens if not tok in stopwords.words('english')]

		return tokens


	def est_sentiment(self, company):
		"""
		Returns the estimated sentiment for a given company, based on news found by self.get_news()
		The resulting sentiment is based on the titles and descriptions of news articles
		"""

		articles = self.get_news(3, company)

		for article in articles:
			article_tokens = self.clean_article(article)

			# Create bag_of_words
			bag_of_words
			for word in article:
				try:
					bag_of_words[word] += 1
				except KeyError:
					bag_of_words[word] = 1










			






