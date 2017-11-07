import numpy as np

def softmax(x,X):
    """
        x: single input scalar from training set
        X: entire training set
    """
    return np.exp(x) / np.sum(np.exp(X))

# some constant value
v = 100
# very negative array
N = np.ones(500) * -v
print(N)
# exptected underflow from 'very negative' valued array when 'v' is ~ 1000
print('\n\n', softmax(N[5],N))

# very positive array
P = np.ones(500) * v
print('\n\n',P)
# expected overflow from 'very positive' valued array when 'v' is ~ 1000
print('\n\n',softmax(P[5],P))

# some constant that would typically cause over/underflow
v = 1000
X = np.random.randn(500) * v
# should resolve underflow/overflow, however, may still result to final value 0
Z = (X) - (np.amax(X))
print('\n\n\n',Z)
print('\n\n\n',softmax(Z[5],Z))
