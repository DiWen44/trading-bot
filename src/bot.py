from dotenv import load_dotenv
import os
import requests
import csv
import json
import re
from datetime import datetime

import finnhub

import SentimentAnalyzer


class Bot():
	"""
	Represents the trading bot

	ATTRIBUTES:
		finnhub_client - The finnhub API client object for this session
		newsapi_key - The API key for newsAPI
		watchlist - An array of 2-value tuples each in form (name, symbol). These are the stocks we will be trading. Loaded from watchlist.csv
		sentiment_analyzer - The sentiment analysis engine, of type SentimentAnalyzer
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

		self.sentiment_analyzer = SentimentAnalyzer("../training_data.csv")


	def __get_headlines(self, days, company):
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

		