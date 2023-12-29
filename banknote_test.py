from naive_bayes import NaiveBayes
import pandas as pd

banknote = pd.read_csv('Banknote/banknote_authentication.csv')

X_banknote = banknote.drop('class', axis=1)
y_banknote = banknote['class']

naive_bayes = NaiveBayes()

naive_bayes.fit(X_banknote, y_banknote)

pred = naive_bayes.predict(X_banknote)

accur = naive_bayes.accuracy_score(pred, y_banknote)

print(accur)
