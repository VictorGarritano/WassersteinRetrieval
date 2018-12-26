from __future__ import division
from wass_funcs import (load_embeddings,
                        clean_corpus_using_embeddings_vocabulary,
                        WassersteinDistances, precisions_at_k)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import sys
import nltk.data
import pandas as pd
from preprocess_twitter import tokenize

import logging

logging.basicConfig(filename='log-{0}-align-{1}-{2}.log'.format(
    sys.argv[6], sys.argv[3], sys.argv[4]), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

print('loading news vectors')
vectors_news = load_embeddings(sys.argv[1], 300)
print('loading tweets vectors')
vectors_tweets = load_embeddings(sys.argv[2], 300)

print("loading dataset")
df = pd.read_csv('whole_dataset_binary_classification.csv')

print('opening news docs')
news = df['article_text'].unique()

print('opening tweet docs')
tweets = df['tweet_text'].apply(lambda row: tokenize(row))

word2keep = sys.argv[5]  # Max words of each document.

print('cleaning news corpus')
clean_news, clean_vectors_news, keys_news = \
    clean_corpus_using_embeddings_vocabulary(word2keep,
                                             set(vectors_news.keys()),
                                             news, vectors_news,
                                             "news",
                                             set(nltk.corpus.stopwords.words(
                                                 "english")))
print('cleaning tweets corpus')
clean_tweets, clean_vectors_tweets, keys_tweets = \
    clean_corpus_using_embeddings_vocabulary(word2keep,
                                             set(vectors_tweets.keys()),
                                             tweets, vectors_tweets,
                                             "tweets",
                                             set(nltk.corpus.stopwords.words(
                                                 "english")))

print("Starting with documents of size", len(clean_news), len(clean_tweets))

del vectors_tweets, vectors_news  # to save space in memory

clean_news, clean_tweets = clean_news.tolist(), clean_tweets.tolist()

print("Starting CountVectorizer")
vec1 = CountVectorizer().fit(clean_news+clean_tweets)

print("Keeping words with associated embeddings")
common = [word for word in vec1.get_feature_names() if word in
          clean_vectors_news or word in clean_vectors_tweets]
W_common = []

print("Keeping words that appear in the corpus")
for w in common:
    if w in clean_vectors_news:
        W_common.append(np.array(clean_vectors_news[w]))
    else:
        W_common.append(np.array(clean_vectors_tweets[w]))
del clean_vectors_news, clean_vectors_tweets

print("The vocabulary size is:", len(W_common))
W_common = np.array(W_common)

print("Generating idf representation")
vect = TfidfVectorizer(vocabulary=common, dtype=np.double, norm=None, )
vect.fit(clean_news+clean_tweets)
X_train_idf = vect.transform(clean_news)
X_test_idf = vect.transform(clean_tweets)

print("Generating tf representation")
vect_tf = CountVectorizer(vocabulary=common, dtype=np.double)
vect_tf.fit(clean_news+clean_tweets)
X_train_tf = vect_tf.transform(clean_news)
X_test_tf = vect_tf.transform(clean_tweets)


valid_idxs = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21,
              22, 23, 24, 25, 26, 28, 29, 30, 32, 34, 35, 36, 37, 39, 40, 41,
              43, 44, 45, 46, 50, 51, 52, 53, 55, 56, 57, 59, 61, 64, 68, 69,
              70, 73, 74, 75, 76, 78, 80, 83, 84, 85, 86, 87, 88, 90, 91, 92,
              93, 94, 97, 98]

print("Starting experiments - idf: Retrieve tweets, given news queries")
clf = WassersteinDistances(W_embed=W_common, n_neighbors=20, n_jobs=16)

mean_aps = []

logging.warning("=======IDF=======")

clf.fit(X_test_idf, np.ones(X_test_idf.shape[0]))

for i in valid_idxs:
    logging.info('STARTING #{0}'.format(i))
    logging.info('getting nearest neighbors')
    dist, preds = clf.kneighbors(X_train_idf[i], n_neighbors=80)
    p5, p10, ap = precisions_at_k(df, news, i, dist, preds)
    logging.info('P@5: {0} \t P@10: {1}'.format(p5, p10))
    mean_aps.append(ap)
logging.info('MAP@10: {0}\n\n'.format(np.mean(np.array(mean_aps))))

print("Starting experiments- tf: Retrieve tweets, given news queries")
clf = WassersteinDistances(W_embed=W_common, n_neighbors=20, n_jobs=16)

mean_aps = []

logging.warning("=======TF=======")

clf.fit(X_test_tf, np.ones(X_test_tf.shape[0]))

for i in valid_idxs:
    logging.info('STARTING #{0}'.format(i))
    logging.info('getting nearest neighbors')
    dist, preds = clf.kneighbors(X_train_tf[i], n_neighbors=80)
    p5, p10, ap = precisions_at_k(df, news, i, dist, preds)
    logging.info('P@5: {0} \t P@10: {1}'.format(p5, p10))
    mean_aps.append(ap)
logging.info('MAP@10: {0}\n\n'.format(np.mean(np.array(mean_aps))))
