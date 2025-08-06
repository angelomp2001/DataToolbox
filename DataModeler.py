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
        models: dict = None,
        features: np.ndarray = None,
        target: np.ndarray = None,
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
        self.train_features = None
        self.train_target = None
        self.valid_features = None
        self.valid_target = None
        self.test_features = None
        self.test_target = None


    def fit(
            self,
            features: np.ndarray,
            target: np.ndarray,
            random_state: int = 12345,
        ):
        '''
        Fits model weights using batch gradient descent for linear/logistic regression using generalized linear model (GLM).
        '''
        verbose = True
        #initialize parameters
        self.features = features
        self.target = target
        if random_state is not None:
            self.random_state = random_state

        X = self.features
        y = self.target

        # define link function (links x to y) 'logit' for binary, 'identity' for continuous. 
        n_unique = np.unique(y).size
        if n_unique == 2:
            # 
            link = 'logit'
        else:
            link = 'identity'    

        ## solve for w by iterating over the df repeatedly
        # initialize losses in loss function
        losses = []    

        # random state seed 
        np.random.seed(self.random_state)

        for _ in range(self.epochs):
            # shuffle rows for each epoch
            idx = np.random.permutation(len(target))
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            # add constant column (intercept) after shuffling
            X_shuffled_with_const = np.concatenate(
                (np.ones((X_shuffled.shape[0], 1)), X_shuffled), axis=1
            )
            
            # initialize w if first epoch (shape depends on feature count after adding const)
            if _ == 0:
                w = np.zeros(X_shuffled_with_const.shape[1], dtype=np.float64) 
            
            # stochastic: divide rows by batch size to get the number of batches
            batches_count = X.shape[0] // self.batch_size

            # go through each batch of rows
            for i in range(batches_count):
                # beginning batch:
                begin = i * self.batch_size
                
                # end batch:
                end = (i + 1) * self.batch_size
                
                # corresponding batch rows:
                X_batch = X_shuffled_with_const[begin:end, :]
                y_batch = y_shuffled[begin:end]

                # Calculate linear predictor (XW)
                mu = np.dot(X_batch, w)

                # select appropriate link function
                if link == 'logit':
                    # mean: g^{-1}(XW) = sigmoid(z)
                    pred = 1 / (1 + np.exp(-mu))

                    # variance: g'^{-1}(XW) = d(sigmoid)/dz = sigmoid(z)*(1 - sigmoid(z))
                    V = np.maximum(pred * (1 - pred), 1e-8)
                else:
                    # mean: g^{-1}(XW) = z Normal (identity) link function (already calculated as z)
                    pred = mu

                    # variance: g'^{-1}(XW) = 1
                    # V = np.ones_like(pred)
                    
                # error term: μ - y
                error = pred - y_batch

                # Calculate and log loss function: L(W)
                if link == 'logit':
                    # To avoid log(0), add a small number (epsilon)
                    eps = 1e-8

                    # L(W) = y*log(μ) + (1-y)*(log(1-μ))
                    loss = -np.mean(y_batch * np.log(pred + eps) + (1 - y_batch) * np.log(1 - pred + eps))
                else:
                    # L(W) = e⊤e
                    loss = np.mean(error ** 2) 
                losses.append(loss)
                
                # print loss
                if verbose and _ % 10 == 0:
                    print(f"Epoch {_}: Loss = {loss:.4f}")


                # ∇_W L(W) = -2X⊤V(y - μ), V = diag[g'^{-1}(XW)], -2 is absorbed in learning rate
                if link == 'logit':
                    # logistic regression: uses variance
                    gradient = np.dot(X_batch.T, error * V) / (X_batch.shape[0] + 1e-8)
                else:
                    # linear regression: variance is 1, so V is removed
                    gradient = np.dot(X_batch.T, error) / (X_batch.shape[0] + 1e-8)
                gradient = np.clip(gradient, -1e5, 1e5)

                # regularization to keep w values low and avoid overfitting (not applied to the constant)
                reg = 2 * w.copy() # remove the 2?
                reg[0] = 0
            
                # update w estimate: W_t = W_{t-1} - 2η X⊤V(y - μ), 2 is absorbed by learning rate
                w -= self.step_size * (gradient + self.reg_weight * reg)

        self.w = np.copy(w[1:])
        self.w0 = float(w[0])
        self.link = link

    def predict(self, test_features):
        return test_features.dot(self.w) + self.w0




# .fit()
# .train(train= df, validate = df, test = df) if none or same provided, then X-validation
# .score(hyperparameters = , objective = 'optimize') if none, score, if provided, score, optimize if objective.
# .pred(features, target)