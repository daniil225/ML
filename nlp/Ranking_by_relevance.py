from nlpia.data.loaders import harry_docs as docs
import copy
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
from collections import OrderedDict
import math

def cosine_sim(vec1, vec2):
    vec1 = [val for val in vec1.values()]
    vec2 = [val for val in vec2.values()]

    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]

    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))

    return dot_prod / (mag_1 * mag_2)


tokenizer = TreebankWordTokenizer()

doc_tokenizer = []

for doc in docs:
    doc_tokenizer += [sorted(tokenizer.tokenize(doc.lower()))]

all_doc_tokens = sum(doc_tokenizer, [])
lexicon = sorted(set(all_doc_tokens))
zero_vector = OrderedDict((token, 0) for token in lexicon)

document_tfidf_vectors = []

for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)

    for key, value in token_counts.items():
        docs_containings_key = 0
        for _doc in docs:
            if key in _doc:
                docs_containings_key += 1
        tf = value / len(lexicon)
        if docs_containings_key:
            idf = len(docs) / docs_containings_key
        else:
            idf = 0
        vec[key] = tf * idf
    document_tfidf_vectors.append(vec)



# Testing
query = "How long does it take to get to the store?"
query_vec = copy.copy(zero_vector)
tokens = tokenizer.tokenize(query.lower())
token_counts = Counter(tokens)
for key, value in token_counts.items():
    docs_containings_key = 0
    for doc_ in docs:
        if key in doc_.lower():
            docs_containings_key += 1
        if docs_containings_key == 0:
            continue
            tf = value / len(tokens)
            idf = len(docs) / docs_containings_key
        query_vec[key] = tf * idf



print(cosine_sim(query_vec, document_tfidf_vectors[0]))
print(cosine_sim(query_vec, document_tfidf_vectors[1]))
print(cosine_sim(query_vec, document_tfidf_vectors[2]))