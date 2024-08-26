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
	- A word2vec vector space mdoel that represents & converts tokenized sentences (from headlines) into vectors
	- A random forest classifier that determines overall sentiment based on these vector embeddings

	ATTRIBUTES:
		sentences_vector_space - a gensim.models.word2vec.Word2Vec object, representing the vector space model
		forest - an sklearn.assemble.RandomForestClassifier, representing the random forest classifier
	"""


	def __init__(self, training_data_csv):
		"""
		CONSTRUCTOR - Creates the word2vec vector space model & the random forest (which is also trained)

		PARAMS:
			training_data_csv - CSV file housing training data
		"""

		# Extract training data from CSV file
		training_sentences = []
		training_sentiments = []
		with open(csv_file_path, newline='') as file:
			reader = csv.reader(file, delimiter=' ')
			for row in reader:
				training_sentiments.append(row[0])

				# Convert headline to 'sentence'
				training_sentences += __headline_to_sentences(row[1])


		# Use word2vec to learn a vector space model for the sentences
		try:
			self.sentences_vector_space = Word2Vec.load("sentences_vector_space")

		except: # Case where no existant model is found - Create new model here
			
			print("NO SENTENCES VECTOR SPACE MODEL FOUND. CREATING NEW MODEL.")
			self.sentences_vector_space = word2vec.Word2Vec(
				sentences=training_sentences,
				vector_size=100,
				min_count=10,
				workers=4
			)
			self.sentences_vector_space.init_sims(replace=True)
			self.sentences_vector_space.save("sentences_vector_space") # Save model

		# Create and train random forest classifier on vectors & yvalues
		self.forest = RandomForestClassifier(n_estimators=100)
		self.forest = forest.fit(training_data_vecs, training_sentiments)


	def __headline_to_sentences(self, headline)
		"""
		PRIVATE METHOD
		Given a headline, convert it to a list of 'sentences' i.e. a list of lists word tokens,
		each sublist represents a single sentence, and each string in it represents a word.
		Essentially this is tokenization.

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

 
	def est_sentiment(self, headlines):
		"""
		Returns the estimated sentiment for a given company, based on provided headlines
		The resulting sentiment is based on the headlines of relevant news articles

		PARAMS:
			headlines - A list of headline strings
		""" 

		# Convert headlines to 'sentences'
		sentences = []
		for headline in headlines:
			sentences += __headline_to_sentences(headline)

		result = self.forest.predict(features)
		return result
