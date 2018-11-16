import pickle
import sklearn
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import math


class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).
        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        n_samples, n_features = X.shape
        w = np.full(n_samples, 1/n_samples)
        # create a new base classifier
        self.weak_classifier = DecisionTreeClassifier()
        for epoch in range(self.n_weakers_limit):
            # update the classifier with the updated sample weight
            self.weak_classifier = self.weak_classifier.fit(X=X, y=y, sample_weight=w)
            w = w.reshape((-1,))
            y = y.tolist()
            predicts = self.predict(X)
            predicts = predicts.tolist()
            e = 0
            # calculate error rate
            for i in range(n_samples):
                if predicts[i] != y[i]:
                    e += w[i]
            e = max(e, 10**(-8))
            # calculate learner confidence
            a = 0.5 * math.log((1 - e) / e)
            s = 0
            # renormalize weights
            for i in range(n_samples):
                w[i] = w[i] * math.exp(-y[i] * a * predicts[i])
                s += w[i]
            for i in range(n_samples):
                w[i] /= s
            y = np.array(y)
        pass

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        pass

# predict the sign of return value
    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        predicts = self.weak_classifier.predict(X)
        predicts = predicts.tolist()
        for i in range(X.shape[0]):
            if predicts[i] > threshold:
                predicts[i] = 1
            else:
                predicts[i] = -1
        predicts = np.array(predicts)
        return predicts

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
