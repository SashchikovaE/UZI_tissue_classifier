#def isPalindrom(str):
#    st = list(str)
#    rev = st.copy()
#    rev.reverse()
#    a = 0
#    for i in range(len(st)):
#       if len(st) % 2 == 1:
#           if i == int(len(st) / 2):
#               continue
#       if rev[i] == st[i]:
#           a += 1
#    if len(st) == a:
#       return True
#
#def split_odd_and_even(str):
#    st = list(str)
#    for i in range(len(st) - 1, 1, -1):
#        _ = st.pop(i)
#        if isPalindrom(st):
#            return st
#        else:
#            return 1
#
#
#
#
#
#
#    #a = 0
#    #for i in range(len(st)):
#    #    if len(st) % 2 == 1:
#    #        if i == int(len(st) / 2):
#    #            continue
#    #    if rev[i] == st[i]:
#    #        a += 1
#    #if len(st) == a:
#    #    return str
#    #if a == 0:
#    #    st.pop()
#    #    return ''.join(st) + ''.join(rev)
#    #co = Counter(st)
#    #g = []
#    #for i, j in co.items():
#    #    if j > 1:
#    #        f = co.index(j)
#    #        g.append(f)
#    #indices = [i for i, x in enumerate(g)]
#
#print(split_odd_and_even("dfgf"))



#нахуй этот кодворс




import numpy as np
import matplotlib.pyplot as plt


def k_means(X, k, iter):
    X = np.array(X, float)
    n, l = X.shape
    indices = np.random.choice(n, k)
    c = X[indices].copy()
    for _ in range(iter):
        distance = np.zeros((n, k))
        for i in range(k):
            distance[:, i] = np.sum((X - c[i]) ** 2, axis=1)
        targets = np.argmin(distance, axis=1)
        new_cs = np.zeros_like(c)
        for j in range(k):
            matrix_j = X[targets == j]
            if len(matrix_j) > 0:
                new_cs[j] = np.mean(matrix_j, axis=0)
            else:
                new_cs[j] = c[j]
        if np.allclose(c, new_cs, atol=1e-5, rtol=1e-3):
            break
    return new_cs, targets

def fit(X, y):
    return np.array(X), np.array(y)
def predict(X, y, k):
    X = np.array(X)
    predictions = [predict_one(X, y, x, k) for x in X]
    return np.array(predictions)

def predict_one(X, y, x, k):
    distances = np.linalg.norm(X - x, axis=1)
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = y[k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common














def fit(X, y):
    return np.array(X, float), np.array(y, float)

def predict(X, y, x, k):
    predictions = [predict_one(X, y, x, k) for x in X]
    return np.array(predictions)

def predict_one(X, y, x, k):
    distance = np.linalg.norm(X - x, axis=1)
    k_near = np.argsort(distance)[:k]
    k_near_label = y[k_near]
    most_common = Counter(k_near_label).most_common(1)[0][0]



from collections import Counter

def do(str):
    s = Counter(str)
    return s



print(do(''))