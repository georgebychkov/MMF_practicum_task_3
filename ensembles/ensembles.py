import time
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


def rooted_mse(y_true, y_pred):
    """
    Computing RMSE
    """
    return np.mean((y_true - y_pred) ** 2) ** 0.5


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.estimators = []
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        feature_subsample_size = X.shape[
            1] // 3 if self.feature_subsample_size is None else self.feature_subsample_size
        train_loss = []
        test_loss = []
        times = []
        train_sum = 0.
        test_sum = 0.
        start = time.perf_counter()
        for _ in range(self.n_estimators):
            inds = np.random.choice(len(y), len(y))
            self.estimators.append(DecisionTreeRegressor(
                max_depth=self.max_depth, max_features=feature_subsample_size, **self.trees_parameters).fit(
                    X[inds], y[inds])
            )
            train_sum += self.estimators[-1].predict(X)
            train_loss.append(rooted_mse(train_sum / len(self.estimators), y))
            times.append(time.perf_counter() - start)
            if X_val is not None:
                test_sum += self.estimators[-1].predict(X_val)
                test_loss.append(rooted_mse(test_sum / len(self.estimators), y_val))
        if X_val is not None:
            return train_loss, test_loss, times
        return train_loss, times

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        return np.array([tree.predict(X) for tree in self.estimators]).mean(0)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.estimators = []
        self.weights = []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        feature_subsample_size = X.shape[1] // 3 if self.feature_subsample_size is None else self.feature_subsample_size
        train_loss = []
        test_loss = []
        times = []
        test_sum = 0
        value = 0
        a = 0
        start = time.perf_counter()
        for _ in range(self.n_estimators):
            gradients = 2*(y - value)
            self.estimators.append(
                DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    max_features=feature_subsample_size,
                    **self.trees_parameters
                ).fit(X, gradients)
            )
            a = minimize_scalar(lambda x: rooted_mse(value + x*self.estimators[-1].predict(X), y)).x
            self.weights.append(self.learning_rate * a)
            value += self.weights[-1] * self.estimators[-1].predict(X)
            times.append(time.perf_counter() - start)
            train_loss.append(rooted_mse(value, y))
            if X_val is not None:
                test_sum += self.weights[-1] * self.estimators[-1].predict(X_val)
                test_loss.append(rooted_mse(test_sum, y_val))
        if X_val is not None:
            return train_loss, test_loss, times
        return train_loss, times

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        result = 0.
        for weight, tree in zip(self.weights, self.estimators):
            result += weight * tree.predict(X)
        return result
