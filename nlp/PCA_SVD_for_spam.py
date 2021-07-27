# First of all load the sms from DataFrame package of nlpia
import pandas as pd
from nlpia.data.loaders import get_data
pd.options.display.width = 120

sms = get_data('sms-spam')
index = ['sms{}{}'.format(i, '!'*j) for (i, j) in zip(range(len(sms)), sms.spam)]
sms.index = index


# Coumpute vector TF-IDF for all mess
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
tfidf_docs = pd.DataFrame(tfidf_docs)
tfidf_docs = tfidf_docs - tfidf_docs.mean() # Centroid vectorized document

# Using PCA for LSA of sms
from sklearn.decomposition import PCA
pca = PCA(n_components=16)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)
columns = ['topic{}'.format(i) for i in range(pca.n_components)]
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=columns, index=index)
print(pca_topic_vectors.round(3).head(6))

column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(), tfidf.vocabulary_.keys())))

weights = pd.DataFrame(pca.components_, columns=terms, index=['topic{}'.format(i) for i in range(16)])
pd.options.display.max_columns = 8
print(weights.head(4).round(3))

pd.options.display.max_columns = 12
deals = weights['! ;) :) half off free crazy deal only $ 80 %'.split()].round(3) * 100
print(deals)
print(deals.T.sum())

# Using tranced SVD for LSA sms
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=16, n_iter=100)
svd_topic_vectros = svd.fit_transform(tfidf_docs.values)
svd_topic_vectros = pd.DataFrame(svd_topic_vectros, columns=columns, index=index)
print(svd_topic_vectros.round(3).head(6))

import numpy as np

svd_topic_vectros = (svd_topic_vectros.T / np.linalg.norm(svd_topic_vectros, axis=1)).T
print(svd_topic_vectros.iloc[:10].dot(svd_topic_vectros.iloc[:10].T).round(1))