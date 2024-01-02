from naive_bayes import NaiveBayes
import pandas as pd


iris = pd.read_csv('Iris/iris.csv')

random_seed = 42 #, random_state=random_seed
random_iris = iris.sample(frac=1).reset_index(drop=True)

X = iris.drop('species', axis=1).values
y = iris['species'].astype('category').cat.codes.values


X_rand = random_iris.drop('species', axis=1).values
y_rand = random_iris['species'].astype('category').cat.codes.values

naive_bayes = NaiveBayes([0, 1, 2])


print("***** Naive Bayes Iris Dataset ****")
print("--------")
print("     ")
print("**** Fit & Predict ****")
print("--------")
print("     ")
naive_bayes.fit(X, y)

pred = naive_bayes.predict(X)

accur = naive_bayes.accuracy_score(pred, y)

print(accur)

print("     ")
print("--------")
print("*** Matrix ****")
print("--------")
print("     ")

confusion = naive_bayes.confusion_matrix(pred, y)

labels = ["Iris setosa", "Iris versicolor","Iris virginica"]

matrix_with_labels = pd.DataFrame(confusion, index=labels, columns=labels)

print(matrix_with_labels)
print("     ")
print("--------")
print("**** N-fold crossvalidation (random) ****")
print("--------")
print("     ")

n_pred, actual = naive_bayes.crossval_predict(X_rand, y_rand, 5)
ac = naive_bayes.accuracy_score(n_pred, actual)
print(ac)
print("     ")
print("--------")

