# instructions for recording functions into a class
# Create a new class
# Copy each function into the class as a method

# In __init__: List common parameters (df) used in any of the related functions
# - Assign these parameters to instance variables (self.xxx)

# per function:
# - add self to function
# - remove default value from self parameters
# - update function parameters that are now instance variables with self.parameter_name
# test method

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataModeler:
    def __init__(
        self,
        models: dict,
        features: np.ndarray,
        target: np.ndarray,
        random_state: int = 12345,
        step_size: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        reg_weight: float = 0.001,
        test_features: np.ndarray = None,
        test_target: np.ndarray = None,
        ):
        self.models = models
        self.features = features
        self.target = target
        self.random_state = random_state
        self.step_size = step_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg_weight = reg_weight
        self.test_features = test_features
        self.test_target = test_target
        self.w = None
        self.w0 = None



    def fit(
            self,
            features: np.ndarray,
            target: np.ndarray,
            
        ):
        '''
        Defines X and Y
        '''
        self.features = features
        self.target = target

        # shuffle rows

        # switch to matrix notation
        # add constant
        X = np.concatenate(
            (np.ones((self.features.shape[0], 1)), self.features), axis=1
        )
        y = self.target
        w = np.zeros(X.shape[1])

        # solve for w by iterating over the df repeatedly
        for _ in range(self.epochs):
            # stochastic: divide rows by batch size to get the number of batches
            batches_count = X.shape[0] // self.batch_size

            # go through each batch of rows
            for i in range(batches_count):
                # beginning batch:
                begin = i * self.batch_size
                
                # end batch:
                end = (i + 1) * self.batch_size
                
                # corresponding batch rows:
                X_batch = X[begin:end, :]
                y_batch = y[begin:end]

                # gradient
                gradient = (
                    2 * X_batch.T.dot(X_batch.dot(w) - y_batch)
                    / X_batch.shape[0]
                )

                # regularization to keep w values low and avoid overfitting
                reg = 2 * w.copy()
                
                # regularization is not applied to the constant
                reg[0] = 0

                # updated gradient, accounting for regularization
                gradient += self.reg_weight * reg
                
                # update w estimate
                w -= self.step_size * gradient

        self.w = w[1:]
        self.w0 = w[0]

    def predict(self, test_features):
        return test_features.dot(self.w) + self.w0




# .fit()
# .train(train= df, validate = df, test = df) if none or same provided, then X-validation
# .score(hyperparameters = , objective = 'optimize') if none, score, if provided, score, optimize if objective.
# .pred(features, target)