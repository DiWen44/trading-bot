import csv
import re
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from gensim.models import word2vec

import nltk
nltk.download() # Download NLTK data
from nltk.corpus import stopwords


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
		
		training_sentiments = []
		training_headlines = []
		training_sentences = [] # Sentences - tokenized & 'cleaned' headlines to train the vec2word model on
		with open(training_data_csv, newline='') as file:
			reader = csv.reader(file, delimiter=' ')
			for row in reader:
				training_sentiments.append(row[0])
				training_headlines.append(row[1])

				training_sentences += self.__headline_to_sentences(row[1])

		print("LOADED TRAINING DATA FROM FILE")


		# Use word2vec to learn a word2vec vector space model for the sentences
		try:
			self.words_vector_space = word2vec.Word2Vec.load("words_vector_space")
			print("USING EXISTING WORD2VEC MODEL.")

		except: # Case where no existant model is found - Create new model here
			
			print("NO WORD2VEC VECTOR SPACE MODEL FOUND. CREATING NEW MODEL.")
			self.words_vector_space = word2vec.Word2Vec(
				sentences=training_sentences,
				vector_size=100,
				min_count=10,
				workers=4
			)
			self.words_vector_space.init_sims(replace=True)
			self.words_vector_space.save("words_vector_space") # Save model

		# Create and train random forest classifier on vectors & yvalues
		self.forest = RandomForestClassifier(n_estimators=100)
		print("CREATED CLASSIFIER")
		training_data_vecs = [ self.__average_feature_vec(headline) for headline in training_headlines ]
		self.forest = self.forest.fit(training_data_vecs, training_sentiments)
		print("TRAINED CLASSIFIER")


	def __headline_to_sentences(self, headline):
		"""
		PRIVATE METHOD
		Given a headline, convert it to a list of 'sentences' i.e. a list of lists word tokens,
		each sublist represents a single sentence, and each string in it represents a word.
		Essentially this is tokenization.

		This is done to generate sentences on which to train the word2vec model

		This method also "cleans" the sentence to prepare for sentiment analysis by:
			- Removing non-letter chars
			- Setting to lowercase
			- Removing stopwords

		returns a cleaned, sentence-split & tokenized version of the original headline

		PARAMS:
			headline - the headline as a string
		"""

		# Tokenize headline string into an array of sentence strings
		# Using NLTK's sentence tokenizer
		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		raw_sentences = tokenizer.tokenize(headline.strip())

		# Clean & tokenize each individual sentence
		sentences = []
		stopword_set = set(stopwords.words('english')) # Convert stopwords to set for constant lookup time
		for raw_sentence in raw_sentences:

			if len(raw_sentence) > 0: # Skip empty sentences

				# Remove non-letters 
				letters_only = re.sub("[^a-zA-Z]", " ", raw_sentence) 

				# Lowercase & tokenize
				tokens = letters_only.lower().split()

				# Remove stopwords
				sentence = [tok for tok in tokens if not tok in stopword_set]

				sentences.append(sentence)

		return sentences


	def __average_feature_vec(self, headline):
		"""
		Get the average feature vector for a headline.
		Since our word2vec vector space model is able to convert individual words to vectors,
		we can get the average vector of all of a headline's constituent words to get a single vector to represent a headline.

		PARAMS:
			headline - The headline as a string
		"""
		words = headline.split() # Tokenize headline to get array of words

		# Set containing all words in vector space model's vocabulary
		model_vocabulary = set(self.words_vector_space.index2word)

		# Initialize resultant feature vec
		result = np.zeros((100),dtype="float32")
		nwords = 0.

		# If word is in the vector space model's vocab, add it's vector to the result
		for word in words:
			if word in model_vocabulary: 
				nwords = nwords + 1.
				result = np.add(self.words_vector_space[word])

		# Divide the result by the number of words to get the average
		result = np.divide(result, nwords)
		return result

 
	def est_sentiment(self, headlines):
		"""
		Returns the estimated sentiment for a given company, based on provided headlines
		The resulting sentiment is based on the headlines of relevant news articles

		PARAMS:
			headlines - A list of headline strings
		""" 

		headline_feature_vec = self.__average_feature_vec(headline)

		result = self.forest.predict(headline_feature_vec)
		print(f"{headline} ------------------- {result}")
		return result
