import sentimentAnalyzer

engine = sentimentAnalyzer.SentimentAnalyzer("training_data.csv")
engine.est_sentiment(["$ESI on lows, down $1.50 to $2.50 BK a real possibility"])
