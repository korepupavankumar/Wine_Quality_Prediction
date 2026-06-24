from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def logistic_model():
    return LogisticRegression(max_iter=1000)


def knn_model():
    return KNeighborsClassifier()


def decision_tree_model():
    return DecisionTreeClassifier()