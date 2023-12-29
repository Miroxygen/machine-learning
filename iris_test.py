from naive_bayes import NaiveBayes
import pandas as pd


iris = pd.read_csv('Iris/iris.csv')

X = iris.drop('species', axis=1) 
y = iris['species']


naive_bayes = NaiveBayes()

naive_bayes.fit(X, y)

pred = naive_bayes.predict(X)

accur = naive_bayes.accuracy_score(pred, y)

print(accur)
