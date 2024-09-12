import sentimentAnalyzer
import bot

bot = bot.Bot()

print(bot.__get_headlines('apple', 3))
del bot # Explicitly call bot's destructor