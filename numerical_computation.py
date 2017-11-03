import numpy as np

X = np.random.randn(100)

O = np.ones(500)*5

print(X)

def softmax(x,X):
    """
        x: input scalar
        X: entire training set
    """
    return np.exp(x) / np.sum(np.exp(X))

print('\n\n\n',softmax(X[2],X))

print('\n\n\n', O)

print('\n\n\n',softmax(O[2],O))
