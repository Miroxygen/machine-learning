import pandas as pd
import numpy as np
import math


class NaiveBayes:
    
  def __init__(self):
    self.attribute_data = {}
    self.category_prob = {}

  def fit(self, X, y):
    self.category_prob = y.value_counts(normalize=True).to_dict()
    for category in self.category_prob:
        category_data = X[y == category]
        self.attribute_data[category] = {
            'mean': np.mean(category_data, axis=0).to_dict(),
            'std': np.std(category_data, axis=0).to_dict()
          }
  
  def gaussian_pdf(self, x, mean, std):
    return -0.5 * np.log(2 * np.pi * std**2) - ((x - mean) ** 2 / (2 * std ** 2))

  def predict(self, data):
    if isinstance(data, pd.Series):
      data = data.to_frame().T
    classifications = []
    for _, new_data_attributes in data.iterrows():
      log_probabilities = {}
      for category, category_data in self.attribute_data.items():
        log_probabilities[category] = np.log(self.category_prob[category])
        for attribute in new_data_attributes.index:
            mean, std = category_data['mean'][attribute], category_data['std'][attribute]
            log_probabilities[category] += self.gaussian_pdf(new_data_attributes[attribute], mean, std)
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
    print(correct_predictions)
    print(total_number_of_predictions)
    return accuracy
  
  def confusion_matrix(self, preds, y):
    confusion = np.zeros((len(self.category_prob), len(self.category_prob)))
    for i, attribute in enumerate(preds):
      if attribute != y[i]:
        print(attribute)
        #confusion.append([attribute, y[i]])
    print(confusion)

