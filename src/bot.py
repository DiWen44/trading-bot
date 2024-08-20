from dotenv import load_dotenv
import os
import requests
import csv

import pandas as pd
import numpy as np

import finnhub



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
		Gets an array of news headlines from newsAPI

		PARAMS:
			days - Number of days prior to get news from (e.g. days=3 will get headlines from the last 3 days)
			company - Name of the company to get news for
		"""

		# Get a JSON object of the news
		news = requests.get(f"https://newsapi.org/v2/everything?q={company}&from=2024-08-20&sortBy=popularity&apiKey={self.newsapi_key}").json()



