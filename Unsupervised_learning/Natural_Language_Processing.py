import os
from nltk import corpus, stem
from nltk.sem.relextract import _join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
from pandas.io.parsers import read_csv
from scipy.sparse import data
from scipy.sparse.construct import random
import seaborn as sns


df = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting = 3)

# cleaning the text 

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # removes conjugation of words like, loved to love so the words are simlified

corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i]) # replace all punctuations (non alphabets) with space
    review = review.lower()                            # converting all to lowercase
    review = review.split()                            # split all the words

    ps = PorterStemmer()

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)] # apply stemming to all the words in review exvept the stop words
    review = ' '.join(review)
    corpus.append(review)

print(corpus)

# creating bag of words 

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray() # saves as array
y = df.iloc[:, -1].values

print(len(X[0]))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
ac = accuracy_score(y_test, y_pred)
print(ac)


