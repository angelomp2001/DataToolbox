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


class DataModeler2:
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
        self.df_vectorized = None
        self.train_features_vectorized = None
        self.train_target_vectorized = None
        self.valid_features_vectorized = None
        self.valid_target_vectorized = None
        self.test_features_vectorized = None
        self.test_target_vectorized = None
        self.link = None
        self.best_params_dict = {}
        self.model_params = {}

    def fit(
        self,
        train_features_vectorized: np.ndarray,
        train_target_vectorized: np.ndarray,
        model_type: str = 'auto',  # 'classification', 'regression'
        model_name: str = None,  # optional: 'xgboost', 'lgbm', 'catboost',
        model_params: dict = None,  # optional
        verbose: bool = False
    ):
        '''
        Fits model weights using a standard model (no internal GLM anymore).
        '''
        # initialize parameters
        self.train_features_vectorized = train_features_vectorized
        self.train_target_vectorized = train_target_vectorized
        X = self.train_features_vectorized
        y = self.train_target_vectorized
        model_params = model_params or {}
        self.model_params = getattr(self, "model_params", {})
        self.model_params.update(model_params)


        # Model registry
        model_map = {
            'classification': {
                'logistic_regression': LogisticRegression,
                'decision_tree': DecisionTreeClassifier,
                'random_forest': RandomForestClassifier,
                'knn': KNeighborsClassifier,
                'xgboost': xgb.XGBClassifier,
                'lgbm': lgb.LGBMClassifier,
                'catboost': cb.CatBoostClassifier,
            },
            'regression': {
                'linear_regression': LinearRegression,
                'random_forest': RandomForestRegressor,
                'xgboost': xgb.XGBRegressor,
                'lgbm': lgb.LGBMRegressor,
                'catboost': cb.CatBoostRegressor,
            }
        }

        # Auto detect model type if not provided
        if model_type == 'auto':
            model_type = 'classification' if np.unique(y).size == 2 else 'regression'

        # Determine default model if not specified
        if model_name is None:
            model_name = list(model_map[model_type].keys())[0]

        # Get model instance
        model_class = model_map[model_type][model_name]
        model = model_class(**self.model_params)


            
        # define link function (links x to y) 'logit' for binary, 'identity' for continuous.
        n_unique = np.unique(y).size
        if n_unique == 2:
            link = 'logit'
        else:
            link = 'identity'

        model.fit(X, y)
        self.fitted_model = model
        self.link = link
        if verbose:
            print(f"Trained {model_name} for {model_type}.")

    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("No model has been fit yet.")

        if return_proba:
            if hasattr(self.fitted_model, "predict_proba"):
                return self.fitted_model.predict_proba(X)[:, 1]
            else:
                raise ValueError("Model does not support probability predictions.")
        return self.fitted_model.predict(X)

    def score(
        self,
        valid_features_vectorized: np.ndarray = None,
        valid_target_vectorized: np.ndarray = None,
        param_to_optimize: str = None,
        param_optimization_range: tuple = None,
        tol: float = 1e-4,
        max_iter: int = 20,
        metric: str = 'auto',
        manual_params: dict = None,
        verbose: bool = True
    ):
        """
        Score the model or optimize a hyperparameter via bisection.

        Parameters:
            - valid_features_vectorized, valid_target_vectorized: validation set
            - param_to_optimize: name of the hyperparameter to optimize
            - param_optimization_range: (low, high) range to search
            - tol: convergence tolerance
            - max_iter: max bisection iterations
            - metric: 'Accuracy', 'Precision', 'Recall', 'F1', 'R2', 'MSE', or 'auto'
            - manual_params: dict of parameters to set before scoring
        """
        # Store validation data internally if provided
        if valid_features_vectorized is not None and valid_target_vectorized is not None:
            self.valid_features_vectorized = valid_features_vectorized
            self.valid_target_vectorized = valid_target_vectorized

        # Load stored validation data if not passed again
        if valid_features_vectorized is None or valid_target_vectorized is None:
            if self.valid_features_vectorized is not None and self.valid_target_vectorized is not None:
                valid_features_vectorized = self.valid_features_vectorized
                valid_target_vectorized = self.valid_target_vectorized
            else:
                raise ValueError("Must provide validation data or set .valid_features_vectorized and .valid_target_vectorized")
        
        # Apply manual parameter overrides
        if manual_params:
            for k, v in manual_params.items():
                setattr(self, k, v)
            if verbose:
                print(f"Manually updated hyperparameters: {manual_params}")
        ################### not self? ################
        def eval_model(features, target):
            self.fit(features, target)
            y_pred = self.predict(valid_features_vectorized)
            y_proba = self.predict(valid_features_vectorized, return_proba=True) if self.link == 'logit' or hasattr(self.fitted_model, "predict_proba") else None

            scores = {
                'Accuracy': accuracy_score(valid_target_vectorized, y_pred),
                'Precision': precision_score(valid_target_vectorized, y_pred, zero_division=0),
                'Recall': recall_score(valid_target_vectorized, y_pred, zero_division=0),
                'F1': f1_score(valid_target_vectorized, y_pred, zero_division=0),
                'MSE': mean_squared_error(valid_target_vectorized, y_pred) if self.link == 'identity' else None,
                'R2': r2_score(valid_target_vectorized, y_pred) if self.link == 'identity' else None,
                'ROC AUC': None,
                'PR AUC': None
            }

            if y_proba is not None:
                try:
                    scores['ROC AUC'] = roc_auc_score(valid_target_vectorized, y_proba)
                    scores['PR AUC'] = average_precision_score(valid_target_vectorized, y_proba)
                except ValueError:
                    pass

            if verbose:
                for metric_name, value in scores.items():
                    print(f"{metric_name}: {value:.4f}" if value is not None else f"{metric_name}: N/A")

            # Return all as default for optimization
            return scores if metric == 'auto' else scores.get(metric, None)

        if not param_to_optimize:
            return eval_model(features=valid_features_vectorized, target=valid_target_vectorized)

        if not param_optimization_range:
            raise ValueError("Must specify param_optimization_range to optimize")

        ## Bisection search for optimal hyperparameter
        # set vars
        low, high = param_optimization_range
        best_score = -np.inf
        best_param = None

        # optimization loop
        for _ in range(max_iter):
            mid = (low + high) / 2
            setattr(self, param_to_optimize, mid)
            score_val = eval_model(features=valid_features_vectorized, target=valid_target_vectorized)

            if verbose:
                print(f"{param_to_optimize} = {mid:.6f}, {metric} = {score_val:.6f}" if score_val is not None else f"{param_to_optimize} = {mid:.6f}, {metric} = N/A")

            if score_val is not None and score_val > best_score:
                best_score = score_val
                best_param = mid
                low, high = low, mid + (high - mid) / 4  # search upper side too
            else:
                high = mid

            if abs(high - low) < tol:
                break

        setattr(self, param_to_optimize, best_param)
        
        # Update best_params_dict
        if not hasattr(self, 'best_params_dict'):
            self.best_params_dict = {}
        self.best_params_dict[param_to_optimize] = best_param

        if verbose:
            print(f"Optimal {{'{param_to_optimize}': {best_param}}}\n{metric}: {best_score}")

        return best_score

    
