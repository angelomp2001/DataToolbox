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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score



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
        validation_x: np.ndarray = None,
        validation_y: np.ndarray = None,
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
        self.validation_x = validation_x
        self.validation_y = validation_y
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
        self.fitted_model = None
        self.link = None



    def _linear_predictor(self, X: np.ndarray) -> np.ndarray:
        '''
        Applies linear transformation and link function to generate predictions.
        '''
        if self.w is None or self.w0 is None:
            raise ValueError("Model has not been fit yet.")
        
        # Add intercept
        X_with_const = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        w_all = np.concatenate(([self.w0], self.w))

        z = np.dot(X_with_const, w_all)

        # Apply link function
        if self.link == 'logit':
            return self._sigmoid(z)
        else:
            return z


    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _log_loss(self, y_true, y_pred, eps=1e-8):
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))


    def fit(
            self,
            features: np.ndarray,
            target: np.ndarray,
            model_type: str = 'auto', # 'classification', 'regression', or 'GLM'
            model_name: str = None, # optional: 'xgboost', 'lgbm', 'catboost',
            verbose: bool = False
        ):
        '''
        Fits model weights using batch gradient descent for linear/logistic regression using generalized linear model (GLM).
        '''
        #initialize parameters
        self.train_features = features
        self.train_target = target
        X = self.train_features
        y = self.train_target
        

        # Model registry
        model_map = {
            'classification': {
                'logistic_regression': LogisticRegression(),
                'decision_tree': DecisionTreeClassifier(),
                'random_forest': RandomForestClassifier(),
                'knn': KNeighborsClassifier(),
                'xgboost': xgb.XGBClassifier(),
                'lgbm': lgb.LGBMClassifier(),
                'catboost': cb.CatBoostClassifier(verbose=0),
            },
            'regression': {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(),
                'xgboost': xgb.XGBRegressor(),
                'lgbm': lgb.LGBMRegressor(),
                'catboost': cb.CatBoostRegressor(verbose=0),
            },
            'GLM': {
                'GLM': 'internal_glm'
            }
        }
        # Auto detect model type if not provided
        if model_type == 'auto':
            model_type = 'classification' if np.unique(y).size == 2 else 'regression'
        
        # Determine default model if not specified
        if model_name is None:
            model_name = list(model_map[model_type].keys())[0]

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

        if model_type == 'GLM' or model_map[model_type][model_name] == 'internal_glm':
            # ========== GLM Training ==========
            if np.unique(y).size == 2:
                # Binary classification
                link = 'logit'
            else:
                # Continuous regression
                link = 'identity'
            
            #self.link = link
            # track losses
            losses = []

            # random state seed 
            np.random.seed(self.random_state)

            for epoch in range(self.epochs):
                # shuffle rows for each epoch
                idx = np.random.permutation(len(target))
                X_shuffled = X[idx]
                y_shuffled = y[idx]

                # add constant column (intercept) after shuffling
                X_shuffled_with_const = np.concatenate(
                    (np.ones((X_shuffled.shape[0], 1)), X_shuffled), axis=1
                )
                
                # initialize w if first epoch (shape depends on feature count after adding const)
                if epoch == 0:
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
                    z = np.dot(X_batch, w)

                    # select appropriate link function
                    if link == 'logit':
                        # mean: g^{-1}(XW) = sigmoid(z)
                        pred = self._sigmoid(z)

                        # variance: g'^{-1}(XW) = d(sigmoid)/dz = sigmoid(z)*(1 - sigmoid(z))
                        V = np.maximum(pred * (1 - pred), 1e-8)
                    else:
                        # mean: g^{-1}(XW) = z Normal (identity) link function (already calculated as z)
                        pred = z

                        # Variance is 1
                        
                    # error term: μ - y
                    error = pred - y_batch

                    # Calculate and log loss function: L(W)
                    if link == 'logit':
                        # To avoid log(0), add a small number (epsilon)
                        # eps = 1e-8

                        # L(W) = y*log(μ) + (1-y)*(log(1-μ))
                        # loss = -np.mean(y_batch * np.log(pred + eps) + (1 - y_batch) * np.log(1 - pred + eps))
                        loss = self._log_loss(y_batch, pred)

                    else:
                        # L(W) = e⊤e
                        loss = np.mean(error ** 2) 
                    losses.append(loss)
                    
                    # print loss
                    if verbose and epoch % 10 == 0:
                        print(f"Epoch {epoch}: Loss = {loss:.4f}")


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
            self.fitted_model = 'GLM'
        else:
            # ========== Standard Model Training ==========
            model = model_map[model_type][model_name]
            model.fit(X, y)
            self.fitted_model = model
            if verbose:
                print(f"Trained {model_name} for {model_type}.")

    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("No model has been fit yet.")
        
        if self.fitted_model == 'GLM':
            preds = self._linear_predictor(X)
            if self.link == 'logit':
                return preds if return_proba else (preds >= 0.5).astype(int)
            else:
                return preds
        else:
            if return_proba:
                if hasattr(self.fitted_model, "predict_proba"):
                    return self.fitted_model.predict_proba(X)[:, 1]
                else:
                    raise ValueError("Model does not support probability predictions.")
            return self.fitted_model.predict(X)

        

    def score(
        self,
        X_validation: np.ndarray = None,
        y_validation: np.ndarray = None,
        optimize: bool = False,
        param_name: str = None,
        param_range: tuple = None,
        tol: float = 1e-4,
        max_iter: int = 20,
        metric: str = 'auto',
        manual_params: dict = None,
        verbose: bool = True
    ):
        """
        Score the model or optimize a hyperparameter via bisection.

        Parameters:
            - X_validation, y_validation: validation set
            - optimize: whether to optimize a hyperparameter
            - param_name: parameter to optimize (e.g. 'step_size', 'reg_weight')
            - param_range: tuple of (low, high) for optimization
            - tol: tolerance for convergence
            - max_iter: max bisection steps
            - metric: 'accuracy', 'r2', 'mse', or 'auto'
            - manual_params: override model params manually before scoring
        """
        if manual_params:
            for k, v in manual_params.items():
                setattr(self, k, v)
            if verbose:
                print(f"Manually updated hyperparameters: {manual_params}")

        if X_validation is None or y_validation is None:
            if self.valid_features is not None and self.valid_target is not None:
                X_validation = self.valid_features
                y_validation = self.valid_target
            else:
                raise ValueError("Must provide validation data or set .test_features and .test_target")            

        def eval_model(features, target):
            self.fit(features, target) # changed from self.features, self.target to features, target
            y_pred = self.predict(X_validation)
            y_proba = self.predict(X_validation, return_proba=True) if self.link == 'logit' or hasattr(self.fitted_model, "predict_proba") else None

            scores = {}
            scores['Accuracy'] = accuracy_score(y_validation, y_pred)
            scores['Precision'] = precision_score(y_validation, y_pred, zero_division=0)
            scores['Recall'] = recall_score(y_validation, y_pred, zero_division=0)
            scores['F1'] = f1_score(y_validation, y_pred, zero_division=0)
            scores['MSE'] = mean_squared_error(y_validation, y_pred) if self.link == 'identity' else None
            scores['R2'] = r2_score(y_validation, y_pred) if self.link == 'identity' else None

            if y_proba is not None:
                try:
                    scores['ROC AUC'] = roc_auc_score(y_validation, y_proba)
                    scores['PR AUC'] = average_precision_score(y_validation, y_proba)
                except ValueError:
                    scores['ROC AUC'] = None
                    scores['PR AUC'] = None
            else:
                scores['ROC AUC'] = None
                scores['PR AUC'] = None

            if verbose:
                for metric_name, value in scores.items():
                    print(f"{metric_name}: {value:.4f}" if value is not None else f"{metric_name}: N/A")

            # Return accuracy as default for optimization
            return scores if metric == 'auto' else scores.get(metric, None) #changed from scores['Accuracy']

        
        if not optimize:
            return eval_model(features=X_validation, target=y_validation) # changed from empty

        if not param_name or not param_range:
            raise ValueError("Must specify param_name and param_range to optimize")

        low, high = param_range
        best_score = -np.inf
        best_param = None

        for _ in range(max_iter):
            mid = (low + high) / 2
            setattr(self, param_name, mid)
            score = eval_model(features=X_validation, target=y_validation) # changed from empty

            if verbose:
                print(f"{param_name} = {mid:.6f}, {metric} = {score:.6f}")

            # Bisection: assume unimodal convex score landscape (roughly)
            if score > best_score:
                best_score = score
                best_param = mid
                low, high = low, mid + (high - mid) / 4  # search upper side too
            else:
                high = mid

            if abs(high - low) < tol:
                break

        setattr(self, param_name, best_param)
        if verbose:
            print(f"Optimal {param_name}: {best_param}, {metric}: {best_score}")

        return best_score



# .train(train= df, validate = df, test = df) if none or same provided, then X-validation
# .score(hyperparameters = , objective = 'optimize') if none, score, if provided, score, optimize if objective.
# .pred(features, target)