import numpy as np
import math


class NaiveBayes:
    
  def __init__(self, classes):
    self.attribute_data = {}
    self.category_prob = {category: 1 / len(classes) for category in classes} 

  def fit(self, X, y):
    for category in self.category_prob:
      category_data = X[y == category]
      mean_values = np.mean(category_data, axis=0)
      std_values = np.sqrt(np.sum((category_data - mean_values) ** 2, axis=0) / (category_data.shape[0] - 1))
      self.attribute_data[category] = {
          'mean': mean_values,
          'std': std_values
      }

  def gaussian_pdf(self, x, mean, std):
    return -0.5 * np.log(2 * np.pi * std**2) - ((x - mean) ** 2 / (2 * std ** 2))

  def predict(self, data):
    if np.ndim(data) == 1:
      data = [data]
    classifications = []
    for attributes in data:
      log_probabilities = {}
      for category, category_data in self.attribute_data.items():
        log_probabilities[category] = np.log(self.category_prob[category])
        for attribute_index in range(len(attributes)):
          mean, std = category_data['mean'][attribute_index], category_data['std'][attribute_index]
          attribute_value = attributes[attribute_index]
          log_probabilities[category] += self.gaussian_pdf(attribute_value, mean, std)
      classifications.append(max(log_probabilities, key=log_probabilities.get))
    return classifications
  
  def accuracy_score(self, preds, y):
    correct_predictions = 0
    total_number_of_predictions = 0
    for i, attribute in enumerate(preds):
      total_number_of_predictions += 1
      if attribute == y[i]:
        correct_predictions += 1
    accuracy = correct_predictions / total_number_of_predictions * 100
    accuracy_data = {"Accuracy" : accuracy, "Correct predictions" : correct_predictions, "Total number of predictions" : total_number_of_predictions}
    return accuracy_data
  
  def confusion_matrix(self, preds, y):
    matrix = np.zeros((len(self.category_prob), len(self.category_prob)), dtype=int)
    for pred, actual in zip(preds, y):
      matrix[pred][actual] += 1
    return matrix

  def crossval_predict(self, X, y, folds):
    fold_accuracies = []
    all_predictions = []
    all_actuals = []
    for i in range(folds):
        training, test = self.get_buckets(X, y, folds, (i + 1))
        self.fit(training["x"], training["y"])
        pred = self.predict(test["x"])
        accuracy = self.accuracy_score(pred, test['y'])
        fold_accuracies.append(accuracy['Accuracy'])
        all_predictions.extend(pred)
        all_actuals.extend(test['y'])
    print(f"Accuracy by fold: {fold_accuracies}")

    return all_predictions, all_actuals
  
  def get_buckets(self, X, y, folds, fold_number):
    bucket_size = int(math.ceil(len(X)) / folds)
    start_index = bucket_size * (fold_number - 1)
    end_index = start_index + bucket_size
    if fold_number == folds:
      end_index = len(X)
    x_test = []
    y_test = []
    x_train = []
    y_train = []
    for i in range(len(X)):
      if start_index <= i < end_index:
        x_test.append(X[i])
        y_test.append(y[i])
      else:
        x_train.append(X[i])
        y_train.append(y[i])
    y_train = np.array(y_train)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    training_data = {"x" : x_train,"y":  y_train}
    test_data = {"x" : x_test, "y" : y_test}
    return training_data, test_data

  

