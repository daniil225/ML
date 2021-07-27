
# Алгоритм VADER для анализа тонаьности
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sa = SentimentIntensityAnalyzer()

corpus = ["Absolutely perfect! Love it! :-) :-) :-)",
          "Horrible! Completely useless. :(",
          "It was OK. Some good and some bad things."]

for doc in corpus:
    scores = sa.polarity_scores(doc)
    #print('{:+}: {}'.format(scores['compound'], doc))



# Наивный байесовский классификатор
from nlpia.data.loaders import get_data
import pandas as pd
from collections import Counter
from nltk.tokenize import casual_tokenize
from sklearn.naive_bayes import MultinomialNB

movies = get_data('hutto_movies')
pd.set_option('display.width', 75)
bags_of_words = []
for text in movies.text:
    bags_of_words.append(Counter(casual_tokenize(text)))


df_bows = pd.DataFrame.from_records(bags_of_words)
df_bows = df_bows.fillna(0).astype(int)

nb = MultinomialNB()
nb = nb.fit(df_bows, movies.sentiment > 0)
movies['predicted_sentiment'] = nb.predict_proba(df_bows)[:, 1] * 8 - 4
movies['error'] = (movies.predicted_sentiment - movies.sentiment).abs()

movies['sentiment_ispositive'] = (movies.sentiment > 0).astype(int)
movies['predicted_ispositive'] = (movies.predicted_sentiment > 0).astype(int)

print((movies.predicted_ispositive == movies.sentiment_ispositive).sum() / len(movies))



products = get_data('hutto_products')
bags_of_words = []
for text in products.text:
    bags_of_words.append(Counter(casual_tokenize(text)))

df_product_bows = pd.DataFrame.from_records(bags_of_words)
df_product_bows = df_product_bows.fillna(0).astype(int)

df_all_bows = df_bows.append(df_product_bows)
df_product_bows = df_all_bows.iloc[len(movies):][df_bows.columns]

products['ispos'] = (products.sentiment > 0).astype(int)
products['predicted_sentiment'] = nb.predict_proba(df_product_bows.values).astype(int)
products['predicted_ispositive'] = (products.predict_sentiment > 0).astype(int)
print((products.predicted_ispositive == products.ispos).sum() / len(products))


