from dotenv import load_dotenv
import os
import requests
import ast
import json
from datetime import datetime, timedelta
import websocket
import pandas as pd
import finnhub

import sentimentAnalyzer


class Bot():
	"""
	Represents the trading bot   

	ATTRIBUTES:

		finnhub_client - A finnhub API client object that allows us to make discretionary requests to finnhub

		finnhub_websocket - A websocket.WebSocketApp object that will allow us to stream real-time price data from finnhub (seperate from the finnhub API client)

		newsapi_key - The API key for newsAPI

		watchlist - A pandas dataframe containing the names and ticker symbols of stocks we will be trading, as well as how much USD worth of that stock the bot is holding,
		 			and the latest recorded price of that stock. Note that this last column does not exist to begin with in the starting watchlist.csv, 
					but will be added to both dataframe and file when this class is instantiated for the first time
					and will be connstantly updated with data from the finnhub websocket

		sentiment_analyzer - The sentiment analysis engine, of type SentimentAnalyzer

		cash - How much cash (USD) the bot currently holds (is added to when shares are sold, and drawn from when shares are bought).
				Stored in cash.txt
	"""


	def __init__(self):

		# Load API keys from .env
		load_dotenv()

		# Load watchlist from watchlist.csv into a dataframe
		self.watchlist = pd.read_csv('watchlist.csv')
		if 'latest_price' not in self.watchlist: # If latest price was not stored in the csv file 
			self.watchlist['latest_price'] = ''

		finnhub_key = os.getenv('FINNHUB_KEY')
		self.finnhub_client = finnhub.Client(api_key=finnhub_key)

		self.finnhub_websocket = websocket.WebSocketApp(f"wss://ws.finnhub.io?token={finnhub_key}",
												  on_open=self.__on_finnhub_websocket_open,
												  on_message=self.__on_finnhub_websocket_message,
												  on_error=self.__on_finnhub_websocket_error,
												  on_close=self.__on_finnhub_websocket_close)

		self.newsapi_key = os.getenv('NEWSAPI_KEY')

		self.sentiment_analyzer = sentimentAnalyzer.SentimentAnalyzer('training_data.csv')

		with open('cash.txt','r') as cash_file: 
			self.cash = int(cash_file.read())
	

	def __del__(self):
		"""
		DESTRUCTOR METHOD
		Saves the watchlist, with it's updated latest price data, back to the watchlist.csv file.
		Also saves updated cash quantity back to cash.txt.
		"""
		
		self.watchlist.to_csv('watchlist.csv', index=False)
		with open('cash.txt','w') as cash_file:
			cash_file.write(str(self.cash))


	def __on_finnhub_websocket_open(self, ws):
		"""
		To be passed to constructor for the finnhub WebSocketApp, when that class is instantiated in bot's constructor.ß
		Subscribes to stocks on watchlist, so that bot gets websocket messages when price changes 
		"""

		print("FINNHUB WEBSOCKET CONNECTION OPENED")

		def subscribe(symbol):
			self.finnhub_websocket.send('{"type":"subscribe","symbol":"' + symbol + '"}')

		self.watchlist['symbol'].apply(subscribe, axis=1)
	

	def __on_finnhub_websocket_message(self, ws, message):
		"""
		To be passed to constructor for the finnhub WebSocketApp, when that class is instantiated in bot's constructor.
		Called automatically when bot receives data from finnhub's websocket server.
		Since in __on_websocket_open() we subscribed to stock prices, this will be called when finnhub sends a price update
		""" 
		
		print(message)
		msg_dict = ast.literal_eval(message)
		symbol = msg_dict['data'][0]['s']
		latest_price = msg_dict['data'][0]['p']
		self.watchlist.loc[self.watchlist['symbol'] == symbol]['latest_price'] = latest_price # Update latest price 

	
	def __on_finnhub_websocket_error(self, ws, error):
		"""
		To be passed to constructor for the finnhub WebSocketApp, when that class is instantiated in bot's constructor.
		Called automatically when a websocket error occurs.
		""" 
		print("FINNHUB WEBSOCKET ERROR:")
		print(error)

	
	def __on_finnhub_websocket_close(self, ws):
		"""
		To be passed to constructor for the finnhub WebSocketApp, when that class is instantiated in bot's constructor.
		Called automatically when the websocket connection is closed
		"""
		print("FINNHUB WEBSOCKET CONNECTION CLOSED")


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


	def __get_candlestick_data(self, symbol, days):
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


	def __trading_strat(self, company, symbol, value_owned, latest_price):
		"""
		Actually executes trades based on collated data.
		Essentially, this method defines the trading strategy with which we approach individual stocks

		PARAMS:
			company - name of the company whose stock is to be traded
			symbol - the company's ticker symbol
			value_owned - The USD worth of company's stock that the bot currently holds
			latest_price - The latest price of the company's stock
		"""

		# Get candlestick data & news over the last day
		candles = self.__get_candlestick_data(symbol, 1)
		headlines = self.__get_headlines(company, 1)

		sentiment = self.sentiment_analyzer.est_sentiment(headlines)


	def trade(self):
		"""
		Executes the trading strategy over all stocks on the watchlist
		"""

		self.watchlist['value_owned'] = self.watchlist.apply(self.__trading_strat, axis=1)
		





		