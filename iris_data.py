import pandas as pd
import numpy as np
import math

iris = pd.read_csv('Iris/iris.csv')

banknote = pd.read_csv('Banknote/banknote_authentication.csv')

grouped_iris = iris.groupby('species')

def create_buckets(data, num_of_buckets):
    bucket_size = math.ceil(len(data) / num_of_buckets)
    buckets = []
    for i in range(0, len(data), bucket_size):
        end_index = min(i + bucket_size, len(data))
        buckets.append(data.iloc[i:end_index])
    return buckets

buckets_by_species = {}

for name, group in grouped_iris:
    buckets_by_species[name] = group


#setosa_buckets = buckets_by_species['Iris-setosa']
#versicolor_buckets = buckets_by_species['Iris-versicolor']
#virginica_buckets = buckets_by_species['Iris-virginica']
    
print(buckets_by_species['Iris-setosa']['sepal_length'][2])
mean_value = np.mean(buckets_by_species['Iris-setosa']['sepal_length'])
print(np.mean(buckets_by_species['Iris-setosa']['sepal_length']))
print(np.std(buckets_by_species['Iris-setosa']['sepal_length']))

