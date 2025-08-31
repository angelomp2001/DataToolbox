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
import tqdm
import time

class DataModeler:
    def __init__(
        self,
        model_type: str = None,  # 'classification', 'regression'
        model_name: str = None,  # optional: 'xgboost', 'lgbm', 'catboost',
        features: np.ndarray = None,
        target: np.ndarray = None,
        random_state: int = 12345,
        step_size: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        reg_weight: float = 0.001,
        valid_features_vectorized: np.ndarray = None,
        valid_target_vectorized: np.ndarray = None,
        test_features: np.ndarray = None,
        test_target: np.ndarray = None,
    ):
        self.model_type = model_type
        self.model_name = model_name
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
        self.fitted_model = None
        self.df_vectorized = None
        self.train_features_vectorized = None
        self.train_target_vectorized = None
        self.valid_features_vectorized = valid_features_vectorized
        self.valid_target_vectorized = valid_target_vectorized
        self.test_features_vectorized = None
        self.test_target_vectorized = None
        self.link = None
        self.best_params_dict = {}
        self.model_params = {}
        self.model_map = {
            'classification': {
                'LogisticRegression': LogisticRegression,
                'DecisionTreeClassifier': DecisionTreeClassifier,
                'RandomForestClassifier': RandomForestClassifier,
                'KNeighborsClassifier': KNeighborsClassifier,
                'xgboost': xgb.XGBClassifier,
                'lgbm': lgb.LGBMClassifier,
                'catboost': cb.CatBoostClassifier,
            },
            'regression': {
                'LinearRegression': LinearRegression,
                'RandomForestRegressor': RandomForestRegressor,
                'xgboost': xgb.XGBRegressor,
                'lgbm': lgb.LGBMRegressor,
                'catboost': cb.CatBoostRegressor,
            }
        }


    def fit(
        self,
        train_features_vectorized: np.ndarray = None,
        train_target_vectorized: np.ndarray = None,
        model_type: str = None,  # 'classification', 'regression'
        model_name: str = None,  # optional: 'xgboost', 'lgbm', 'catboost',
        model_params: dict = None,  # optional
        verbose: bool = False
    ):
        '''
        Fits model weights based on model type and parameters.
        If model_type is 'auto', it will determine based on target variable.
        '''
        # initialize parameters
        if train_features_vectorized is not None and train_target_vectorized is not None:
            self.train_features_vectorized = train_features_vectorized
            self.train_target_vectorized = train_target_vectorized
        else:
            if self.features is not None and self.target is not None:
                self.train_features_vectorized = self.features
                self.train_target_vectorized = self.target
            else:
                raise ValueError("Must provide training data or set .features and .target")
        
        X = self.train_features_vectorized
        y = self.train_target_vectorized
        model_params = model_params or {}
        self.model_params = getattr(self, "model_params", {})
        self.model_params.update(model_params)


        # Model registry
        model_map = self.model_map

        # Auto detect model type if not provided
        if model_type is not None:
            if model_type in model_map:
                self.model_type = model_type
            else:
                raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(model_map.keys())}.")
        else:
            if self.model_type is not None:
                model_type = self.model_type
            else:
                # Auto-detect model type based on target variable
                model_type = 'classification' if np.unique(y).size == 2 else 'regression'

        # Determine default model if not specified
        if model_name is not None and not isinstance(model_name, list):
            if model_name in model_map[model_type]:
                self.model_name = model_name
        elif model_name is None:
            if self.model_name is not None:
                model_name = self.model_name
            else:
                print(f"Model name not specified, using default for {model_type}.")

        # set model
        if isinstance(model_name, list):
            for name in model_name:
                if name in model_map[model_type]:
                    self.model_name = name
                    model_class = model_map[model_type][name]
                    model = model_class(**self.model_params)
                    print(f"Training {model}")
                    # Fit the model
                    start_time = time.time()
                    self.fitted_model = model.fit(X, y) 
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Training time: {elapsed_time:.2f} seconds")

                    if verbose:
                        print(f"Trained {name} for {model_type}.")

        else:
            # Setting model to the first one, or the specific one if mentioned
            if model_name in model_map[model_type]:
                # save model name to self for use in score method
                self.model_name = model_name
                model_class = model_map[model_type][model_name]
                model = model_class(**self.model_params)
                
                # Fit the model
                start_time = time.time()
                self.fitted_model = model.fit(X, y) 
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Training time: {elapsed_time:.2f} seconds")

                if verbose:
                    print(f"Trained {model_name} for {model_type}.")

            else:
                raise ValueError(f"Invalid model_name: {model_name}. Must be one of {list(model_map[model_type].keys())}.")
        

    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("No model has been fit yet.")

        if return_proba:
            if hasattr(self.fitted_model, "predict_proba"):
                start_time = time.time()
                y_pred_proba = self.fitted_model.predict_proba(X)[:, 1]
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Prediction time: {elapsed_time:.2f} seconds")
                return y_pred_proba
            else:
                raise ValueError("Model does not support probability predictions.")
        return self.fitted_model.predict(X)

    def score(
        self,
        valid_features_vectorized: np.ndarray = None,
        valid_target_vectorized: np.ndarray = None,
        param_to_optimize: str = None,
        param_optimization_range: tuple = None,
        tolerance: float = 1e-4,
        max_iter: int = 20,
        metric: str = None, # 'Accuracy', 'Precision', 'Recall', 'F1', 'R2', 'RMSE'
        manual_params: dict = None,
        verbose: bool = False
    ):
        """
        Score the model or optimize a hyperparameter via bisection.

        Parameters:
            - valid_features_vectorized, valid_target_vectorized: validation set
            - param_to_optimize: name of the hyperparameter to optimize
            - param_optimization_range: (low, high) range to search
            - tolerance: convergence tolerance
            - max_iter: max bisection iterations
            - metric: 'Accuracy', 'Precision', 'Recall', 'F1', 'R2', 'RMSE', ROC AUC', 'PR AUC'
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
                # Update model_params with the new manual parameters
                self.model_params[k] = v
                
                # Refit the model with updated parameters
                model_class = self.model_map[self.model_type][self.model_name]
                
                # Create a new model instance with updated parameters
                model = model_class(**self.model_params)

                # Fit the model with updated parameters
                self.fitted_model = model.fit(self.train_features_vectorized, self.train_target_vectorized)

            if verbose:
                print(f"Manually updated hyperparameters: {manual_params}")
        
        def eval_model(
                features: np.ndarray,
                target: np.ndarray,
                ):
            #self.fit(features, target)
            start_time = time.time()
            y_pred = self.predict(features)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Prediction time: {elapsed_time:.2f} seconds")
            print(f'y_pred: {y_pred.shape}')
            y_proba = self.predict(valid_features_vectorized, return_proba=True) if self.model_type == 'classification' else None
            
            scores = {}

            try:
                scores['Accuracy'] = accuracy_score(valid_target_vectorized, y_pred)
            except Exception as e:
                scores['Accuracy'] = f"Error: {e}"

            try:
                scores['Precision'] = precision_score(valid_target_vectorized, y_pred, zero_division=0)
            except Exception as e:
                scores['Precision'] = f"Error: {e}"

            try:
                scores['Recall'] = recall_score(valid_target_vectorized, y_pred, zero_division=0)
            except Exception as e:
                scores['Recall'] = f"Error: {e}"

            try:
                scores['F1'] = f1_score(valid_target_vectorized, y_pred, zero_division=0)
            except Exception as e:
                scores['F1'] = f"Error: {e}"

            try:
                scores['RMSE'] = np.sqrt(mean_squared_error(valid_target_vectorized, y_pred))
            except Exception as e:
                scores['RMSE'] = f"Error: {e}"

            try:
                scores['R2'] = r2_score(valid_target_vectorized, y_pred)
            except Exception as e:
                scores['R2'] = f"Error: {e}"

            # Placeholder for ROC AUC and PR AUC if applicable
            scores['ROC AUC'] = None
            scores['PR AUC'] = None

            if y_proba is not None:
                try:
                    scores['ROC AUC'] = roc_auc_score(valid_target_vectorized, y_proba)
                    scores['PR AUC'] = average_precision_score(valid_target_vectorized, y_proba)
                except ValueError:
                    pass

            if verbose:
                for metric_name, value in scores.items():
                    print(f"{metric_name}: {value}" if value is not None else f"{metric_name}: N/A")

            # Return all as default for optimization
            if metric is None:
                for metric_name, value in scores.items():
                    print(f"{metric_name}: {value}" if value is not None else f"{metric_name}: N/A")
                return scores
            print(f'{metric}: {scores.get(metric, None)}')
            return scores.get(metric, None)

        if not param_to_optimize:
            
            return eval_model(features=valid_features_vectorized, target=valid_target_vectorized)

        if not param_optimization_range or metric is None:
            raise ValueError("Must specify param_optimization_range to optimize and a metric to optimize against.")

        ## Bisection search for optimal hyperparameter
        # set vars
        low, high = param_optimization_range
        best_score = -np.inf
        best_param = None

        # optimization loop
        for _ in tqdm(
            range(max_iter),
            desc = 'Bisection Search',
            unit = 'iteration',
            leave = False,
            colour = 'green',
            ascii = (" ", "█"),
            ncols = 100
        ):
            if self.model_type == 'classification':
                mid = (low + high) / 2
                #setattr(self, param_to_optimize, mid)
                self.model_params[param_to_optimize] = int(mid)

                model_class = self.model_map[self.model_type][self.model_name]
                model = model_class(**self.model_params)
                self.fitted_model = model.fit(self.train_features_vectorized, self.train_target_vectorized)
                score_val = eval_model(features=valid_features_vectorized, target=valid_target_vectorized)

            if verbose:
               print(f"{param_to_optimize} = {mid:.6f}, {metric} = {score_val:.6f}" if score_val is not None else f"{param_to_optimize} = {mid:.6f}, {metric} = N/A")

            if score_val is not None and score_val > best_score:
                best_score = score_val
                best_param = mid
                low, high = low, mid + (high - mid) / 4  # search upper side too
            else:
                high = mid

            if abs(high - low) < tolerance:
                break

        setattr(self, param_to_optimize, best_param)
        
        # Update best_params_dict
        if not hasattr(self, 'best_params_dict'):
            self.best_params_dict = {}
        self.best_params_dict[param_to_optimize] = best_param

        #if verbose:
        print(f"Optimal {{'{param_to_optimize}': {best_param}}}\n{metric}: {best_score}")

        return best_score

    
