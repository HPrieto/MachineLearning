import numpy as np
from math import sqrt
import warnings
from matplotlib import style
from collections import Counter
import pandas
import random


def k_nearest_neigbors(data, predict, k=5):
    """
        Data:    Data Model Destination
        Predict: Dataset
        K:       N Nearest Neighbors to check
    """
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidian_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidian_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence


# Import data
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# Replace Bad Data
df.replace('?', -99999, inplace=True)

# Drop useless columns
df.drop(['id'], 1, inplace=True)

# Convert values to float
full_data = df.astype(float).values.tolist()

# Shuffle Data
random.shuffle(full_data)

test_size = 0.4

# Empty train set, holds good or bad cancer features
train_set = {2: [], 4: []}

# Empty test set, holds good or bad cancer features
test_set = {2: [], 4: []}

# Hold first 80% of data
train_data = full_data[:-int(test_size * len(full_data))]

# Hold last 20% of data
test_data = full_data[-int(test_size * len(full_data)):]

# Populate dictionaries
for i in train_data:
    # Identify which set and add
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    # Identify which set and add
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

# Iterate through class groups in data
for group in test_set:
    # Iterate through features per set
    for data in test_set[group]:
        vote, confidence = k_nearest_neigbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy: ', correct / total)
