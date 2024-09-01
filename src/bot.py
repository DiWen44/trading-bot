from dotenv import load_dotenv
import os
import requests
import csv
import json
import re
from datetime import datetime, timedelta
import pandas as pd
import finnhub

import sentimentAnalyzer


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

		self.sentiment_analyzer = sentimentAnalyzer.SentimentAnalyzer("../training_data.csv")


	def __get_headlines(self, company, days):
		"""
		Gets news headlines from newsAPI that are pertinent to a given company.
		Returned in the form of an array of strings, each string representing the headline of an article

		PARAMS:
			company - Name of the company to get news headlines for
			days - Number of days prior to get news headlines from (e.g. days=3 will get headlines from the last 3 days)
		"""
		
		# Get start date of time window in string form
		start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')

		# Get a dictionary object of the news (newsAPI returns a JSON object)
		newsJSON = requests.get(f"https://newsapi.org/v2/everything?q={company}&from={start_date}&sortBy=popularity&apiKey={self.newsapi_key}").json()
		news = json.loads(newsJSON)

		headlines = [ (article['title'] + ". " + article['description']) for article in news['articles'] ] # NewsAPI data seperates headlines into title and description - concatenate these 2 together for our purposes
		return headlines


	def __get_finnhub_data(self, symbol, days):
		"""
		Gets stock price candlestick data from finnhub for a given company.
		The candles' interval will be in days, so each candlestick represents the stock over a day (rather than a week or month) 
		Returns this data as a pandas dataframe of the finnhub API response

		PARAMS:
			symbol - Ticker symbol of the company to get data for
			days - Time window (in days) in which to get data (e.g. days=3 will get timeseries data over the last 3 days)
		"""

		present_date = datetime.today() 
		window_start_date = present_date - timedelta(days=days)

		candles = self.finnhub_client.stock_candles(symbol, 'D', present_date.timestamp(), window_start_date.timestamp())
		return pd.DataFrame(candles)
	

	def trading_strat(self, candle_data, sentiment):
		"""
		Actually executes trades based on collated data.
		Essentially, this method defines the trading strategy with which we approach individual stocks

		PARAMS:
			candle_data - A pytorch dataframe representing the candlestick data for a single stock. This is the data
							returned by the __get_finnhub_data method in this class
			sentiment - The estimated sentiment of this stock, as computed by the sentiment analysis engine
		"""

		