from naive_bayes import NaiveBayes
import pandas as pd

#Data

banknote = pd.read_csv('Banknote/banknote_authentication.csv')
banknote_random = banknote.sample(frac=1).reset_index(drop=True)

random_seed = 42 #, random_state=random_seed
X_banknote = banknote.drop('class', axis=1).values
y_banknote = banknote['class']

x_rand = banknote_random.drop('class', axis=1).values
y_rand = banknote_random['class']

#Start

naive_bayes = NaiveBayes([0, 1])

#Fit and predict

naive_bayes.fit(x_rand, y_rand)

pred = naive_bayes.predict(x_rand)

accur = naive_bayes.accuracy_score(pred, y_rand)

#Matrix

matrix = naive_bayes.confusion_matrix(pred, y_rand)

labels = ["Real", "False"]

matrix_with_labels = pd.DataFrame(matrix, index=labels, columns=labels)

#Print

print("***** Naive Bayes Banknote Dataset ****")
print("--------")
print("     ")
print("**** Fit & Predict ****")
print("--------")
print("     ")
print(accur)
print("     ")
print("*** Matrix ****")
print("     ")
print(matrix_with_labels)
print("     ")
print("--------")
print("**** N-fold crossvalidation (random) ****")
print("     ")

#Crossval

n_pred, n_acutal = naive_bayes.crossval_predict(x_rand, y_rand, 5)
n_accur = naive_bayes.accuracy_score(n_pred, n_acutal)
print(n_accur)
print("     ")