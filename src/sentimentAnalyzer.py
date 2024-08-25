import csv

from sklearn.ensemble import RandomForestClassifier
from gensim.models import word2vec

import nltk
nltk.download() # Download NLTK data
from nltk.corpus import stopwords


class SentimentAnalyzer():
	"""
	Represents the bot's sentiment analysis engine

	The sentiment analysis engine consists of 1 parts:
	- A word2vec vector space mdoel that represents & converts words into vectors
	- A random forest classifier that determines overall sentiment based on these vector embeddings

	ATTRIBUTES:
		vector_space - a gensim.models.word2vec object, representing the vector space model
		forest - an sklearn.assemble.RandomForestClassifier, representing the random forest classifier
	"""


	def __init__(self, training_data_csv):
		"""
		CONSTRUCTOR - Creates the word2vec vector space model & the random forest (which is also trained)

		PARAMS:
			training_data_csv - CSV file housing training data
		"""

		# Extract training data from CSV file
		with open(csv_file_path, newline='') as file:
			reader = csv.reader(file, delimiter=' ')
			training_data = [row for row in reader]

		# Make new array of clean headlines in training data
		clean_training_headlines = [__clean_headline(row[1]) for row in training_data]

		# Also create an array of y-values/sentiments (i.e. positive, neutral, negative) from the training data
		yvalues = [row[0] for row in training_data] 

		# Use word2vec to learn a vector space model for the headlines
		self.vector_space = word2vec.Word2Vec(
			sentences=clean_training_headlines,
			vector_size=100,
			min_count=1,
			workers=4
		)

		# Create and train random forest classifier on vectors & yvalues
		self.forest = RandomForestClassifier(n_estimators=100)
		self.forest = forest.fit(training_data_vecs, yvalues)


	def __clean_headline(self, headline)
		"""
		PRIVATE METHOD
		Given a headline, "clean" it to prepare for sentiment analysis by:
			- Removing non-letter chars
			- Setting to lowercase
			- Tokenizing
			- Removing stopwords

		returns a cleaned & tokenized - as in an array of string tokens/words - version of the original string

		PARAMS:
			headline - the headline as a string
		"""

		letters_only = re.sub("[^a-zA-Z]", " ", text) # Remove non-letters 
		tokens = letters_only.lower().split() # Lowercase & tokenize

		# Remove stopwords
		stopword_set = set(stopwords.words('english')) # Convert to set for constant lookup time
		meaningful_word_tokens = [tok for tok in tokens if not tok in stopword_set]

		return meaningful_word_tokens

 
	def est_sentiment(self, company):
		"""
		Returns the estimated sentiment for a given company, based on news found by self.get_headlines()
		The resulting sentiment is based on the headlines of relevant news articles

		PARAMS:
			company - Name of the company
		""" 

		# Get headlines, then 'clean' them
		headlines = self.get_headlines(3, company)
		clean_headlines = []
		for headline in headlines:
			cleaned_headline = self.clean_article(headline)
			clean_headlines.append(cleaned_headline)


		result = self.forest.predict(features)
