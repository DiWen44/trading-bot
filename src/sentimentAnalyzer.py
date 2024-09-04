import re
from os import path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from gensim.models import word2vec

import nltk
nltk.download_shell() # Download NLTK data
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


class SentimentAnalyzer():
	"""
	Represents the bot's sentiment analysis engine

	The sentiment analysis engine consists of 1 parts:
	- A word2vec vector space model that represents & converts individual words (from headlines) into vectors
	- A random forest classifier that determines overall sentiment based on a vector embedding for a headline. 
		This vector embedding is created by averaging the vectors that represent the headline's constituent words

	ATTRIBUTES:
		words_vector_space - a gensim.models.word2vec.Word2Vec object, representing the vector space model
		forest - an sklearn.assemble.RandomForestClassifier, representing the random forest classifier
	"""


	def __init__(self, training_data_csv):
		"""
		CONSTRUCTOR - Creates the word2vec vector space model & the random forest (which is also trained)

		PARAMS:
			training_data_csv - CSV file housing training data
		"""

		# Extract training data from CSV file
		training_data = pd.read_csv(training_data_csv)

		# Get sentences pool from headlines - to train word2vec model
		sentences_pool = []
		for headline in training_data['headline']:
			headline_sentences = self.__headline_to_sentences(headline)
			sentences_pool += headline_sentences

		print("LOADED TRAINING DATA FROM FILE")

		# Use word2vec to learn a word2vec vector space model for the sentences
		if path.isfile("words_vector_space"):
			# If a model has already been created and saved
			self.words_vector_space = word2vec.Word2Vec.load("words_vector_space")
			print("USING EXISTING WORD2VEC MODEL.")

		else: 
			# Case where no existant model is found - Create new model here
			print("NO WORD2VEC VECTOR SPACE MODEL FOUND. CREATING NEW MODEL.")
			self.words_vector_space = word2vec.Word2Vec(
				sentences=sentences_pool,
				vector_size=100,
				min_count=10,
				workers=4
			)
			self.words_vector_space.init_sims(replace=True)
			self.words_vector_space.save("words_vector_space") # Save model

		# Create and train random forest classifier on vectors & yvalues
		self.forest = RandomForestClassifier()
		print("CREATED CLASSIFIER")

		training_data['vector'] = training_data.apply(self.__average_feature_vec, axis=1) # Get feature vectors for each headline in training data
		
		print(training_data[['vector', 'sentiment']])
		self.forest = self.forest.fit(training_data['vector'].to_list(), training_data['sentiment'].to_list())
		print("TRAINED CLASSIFIER")


	def __headline_to_sentences(self, headline):
		"""
		PRIVATE METHOD
		Convert a headline to a list of the 'sentences' that constitute it (a sentence being an array of word tokens that make up a grammatical sentence)
		In the returned list, each sublist represents a single sentence, and each string in it represents a word.
		Essentially this is tokenization.

		This is done to generate sentences on which to train the word2vec model

		This method also "cleans" the sentence to prepare for sentiment analysis by:
			- Removing non-letter chars
			- Setting to lowercase
			- Removing stopwords

		Returns a cleaned, sentence-split & tokenized version of the original headline

		PARAMS:
			headline - the headline, as a string
		"""
		# Tokenize headline string into an array of sentence strings
		# Using NLTK's sentence tokenizer
		raw_sentences = sent_tokenize(headline.strip())

		# Clean & tokenize each individual sentence
		sentences = []
		stopword_set = set(stopwords.words('english')) # Convert stopwords to set for constant lookup time
		for raw_sentence in raw_sentences:

			if len(raw_sentence) > 0: # Skip empty sentences

				# Remove non-letters - substitute them with spaces
				letters_only = re.sub("[^a-zA-Z]", " ", raw_sentence) 

				# convert all letters to lowercase & tokenize
				tokens = word_tokenize(letters_only.lower())

				# Remove stopwords
				sentence = [tok for tok in tokens if not tok in stopword_set]

				sentences.append(sentence)

		return sentences


	def __average_feature_vec(self, row):
		"""
		Get the average feature vector for a headline.
		Since our word2vec vector space model is able to convert individual words to vectors,
		we can get the average vector of all of a headline's constituent words to get a single vector to represent a headline.

		PARAMS:
			row - A pd.series representing a row of the dataframe of training data from the CSV file.
					This contains the headline.
					Note that we must pass the entire row, so that this method can be passed to df.apply()
		"""

		headline = row['headline']

		words = headline.split() # Tokenize headline to get array of words

		# Set containing all words in vector space model's vocabulary
		model_vocabulary = set(self.words_vector_space.wv.index_to_key)

		# Initialize resultant feature vec
		result = np.zeros((100),dtype="float32")
		numWords = 0.

		# If word is in the vector space model's vocab, add it's vector to the result
		for word in words:
			if word in model_vocabulary: 
				numWords = numWords + 1.
				result = np.add(result, self.words_vector_space.wv[word])

		# Divide the result by the number of words to get the average
		result = np.divide(result, numWords)
		return result


	def est_sentiment(self, headline):
		"""
		Returns the estimated sentiment for a headline
		The resulting sentiment is based on the headlines of relevant news articles

		PARAMS:
			headlines - A headline string
		""" 

		# get feature vector representing entire headline, 
		headline_feature_vec = self.__average_feature_vec(headline)

		result = self.forest.predict(headline_feature_vec)
		print(f"{headline} ------------------- {result}")
		return result
