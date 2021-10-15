# E.coli dataset, also referred to as the “protein localization sites” dataset
# imbalanced multiclass classification
# "https://archive.ics.uci.edu/ml/datasets/ecoli"

from operator import imod
import os
from typing import Counter
import pandas as pd
import matplotlib.pyplot as plt
from pandas._config.config import set_option
from pandas.core.indexes.multi import maybe_droplevels
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from statistics import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline, make_pipeline 
from imblearn.over_sampling import SMOTE



# define the dataset location & load the csv file as a data frame
filename = 'ecoli.csv'
df = pd.read_csv(filename, header=None)
df = df[df[7] != 'imS']
df = df[df[7] != 'imL']
x = df.iloc[:,:-1].values
y = df.iloc[:, -1].values # summarize the class distribution
le = LabelEncoder()
y = le.fit_transform(y)
# or use --> y = LabelEncoder().fit_transform(y)

#print(df.shape)
set_option('precision', 3)
print(df.describe())
#######################################################################
### or You can use the function defined (load_dataset) below to do the same

counter = Counter(y)
for k,v in counter.items(): # to identify the class and the # in the class
    per = v/len(y)*100      # percentage = # in class/len(y) which is 336 in this case x 100
    print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))
df.hist(bins=25)
plt.show()

# function to load the dataset and split the input variables into inputs and output variables and use a label encoder to ensure class labels are numbered sequentially
def load_dataset (full_path):
    df1 = pd.read_csv('/home/varikord/Projects/python_scripts/MACHINE_LEARNING/machinelearningmastery/ecoli.csv', header=None)
    # to remove datavalues from the two classes with lower #
    df1 = df1[df1[7] != 'imS']
    df1 = df1[df1[7] != 'imL']
    data = df1.values
    x, y = data[:, :-1], data[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)
    return x, y, le

# evaluation of the model
def evaluate_model(x, y, model):
    cv = RepeatedStratifiedKFold(n_repeats=3, n_splits=5, random_state=1) # define evaluation procedure
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1) # evaluation
    return scores

'''
print(x.shape, y.shape, Counter(y))
# defining a dummy classifier to be used as a reference model
model = DummyClassifier(strategy='most_frequent')
scores = evaluate_model(x, y, model)
print('Mean accuracy: %.3f (%.3f)' % (mean(scores), stdev(scores) ))

'''

# define various ML models to test
def test_models():
    models, names = list(), list()
    #Linear Discriminant Analysis (LDA)
    models.append(LinearDiscriminantAnalysis())
    names.append('LDA')
    # Logistic regression
    models.append(LogisticRegression(solver='lbfgs', multi_class='multinomial'))
    names.append('LR')
    #Support Vector Machine (SVM)
    models.append(LinearSVC())
    names.append('SVM')
    #Bagged Decision Trees (BAG)
    models.append(BaggingClassifier(n_estimators=1000))
    names.append('BAG')
    #Random Forest (RF)
    models.append(RandomForestClassifier(n_estimators=1000))
    names.append('RF')
    #Extra Trees (ET)
    models.append(ExtraTreesClassifier(n_estimators=1000))
    names.append('ET')
    #KN classifier
    models.append(KNeighborsClassifier(n_neighbors=3))
    names.append('KNC')
    #Gaussian process classifier
    models.append(GaussianProcessClassifier())
    names.append('GPC')
    return models, names


# run and evaluate every model in the func test_models

models, names = test_models()
results = list()
for i in range(len(models)):
    steps = [('o', SMOTE(k_neighbors=2)), ('m', models[i])]
    pipeline = Pipeline(steps=steps)
    #scores = evaluate_model(x, y, models[i])
    scores = evaluate_model(x, y, pipeline)
    results.append(scores)
    #o/p results
    print('>%s %.3f (%.3f)' % (names[i], mean(scores), stdev(scores)))


plt.boxplot(results, labels=names,showmeans=True)
plt.show()

# evaluation vs prediction

model = RandomForestClassifier(n_estimators=1000)
model.fit(x, y)

# known class "cp"
row = [0.49,0.29,0.48,0.50,0.56,0.24,0.35]
yhat = model.predict([row])
label = le.inverse_transform(yhat)[0]
print('>Predicted=%s (expected cp)' % (label))




