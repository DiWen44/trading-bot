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

		newsapi_key - The API key for newsAPI

		watchlist - A pandas dataframe containing the names and ticker symbols of stocks we will be trading, as well as the quantity of stock the bot is holding (owned).

		sentiment_analyzer - The sentiment analysis engine, of type SentimentAnalyzer

		cash - How much cash (USD) the bot currently holds (is added to when shares are sold, and drawn from when shares are bought).
				Stored in cash.txt

		cash_at_risk - Proportion of cash that is allowed to be risked on one trade. given by (maximum cash sum to risk at once)/(total cash)
						(e.g. if account_risk is 0.2, then 20% of the cash can be spent on one trade)
	"""


	def __init__(self, cash_at_risk):

		# Load API keys from .env
		load_dotenv()

		# Load watchlist from watchlist.csv into a dataframe
		self.watchlist = pd.read_csv('watchlist.csv')

		finnhub_key = os.getenv('FINNHUB_KEY')
		self.finnhub_client = finnhub.Client(api_key=finnhub_key)
		
		self.newsapi_key = os.getenv('NEWSAPI_KEY')

		self.sentiment_analyzer = sentimentAnalyzer.SentimentAnalyzer('training_data.csv')

		with open('cash.txt','r') as cash_file: 
			self.cash = int(cash_file.read())

		self.cash_at_risk = cash_at_risk
	

	def __del__(self):
		"""
		DESTRUCTOR METHOD
		Saves the watchlist, with it's updated stock holding info, back to the watchlist.csv file.
		Also saves updated cash quantity back to cash.txt.
		"""
		
		self.watchlist.to_csv('watchlist.csv', index=False)
		with open('cash.txt','w') as cash_file:
			cash_file.write(str(self.cash))
	
	
	def __get_price(self, symbol):
		"""
		Gets a company's latest recorded share price from finnhub

		PARAMS:
			symbol - The company's ticker symbol
		"""
		return finnhub.quote(symbol)['c']

 

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
		news = requests.get(f"https://newsapi.org/v2/everything?q={company}&searchin=title&language=en&from={start_date}&sortBy=popularity&apiKey={self.newsapi_key}").json()

		headlines = [ (article['title'] + ". " + article['description']) for article in news['articles'] if article['description'] is not None] # NewsAPI data seperates headlines into title and description - concatenate these 2 together for our purposes
		return headlines


	def trading_strat(self, symbol):
		"""
		Actually executes trades based on collated data.
		Essentially, this method defines the trading strategy with which we approach individual stocks

		PARAMS:
			symbol - the ticker symbol of the company whose stock to trade
		"""

		# Get news sentiment over the last day
		company_name = self.watchlist.loc[self.watchlist['symbol'] == symbol]['name']
		headlines = self.__get_headlines(company_name, 1)
		sentiment = self.sentiment_analyzer.est_sentient(headlines)

		price = self.__get_price(symbol)

		# Buy if sentiment positive, sell if negative
		# If sentiment neutral, do nothing
		if sentiment == 'positive':
			
		elif sentiment == 'negative':
			

		





		