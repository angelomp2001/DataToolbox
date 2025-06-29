# This code is part of a machine learning pipeline that selects the best model and optimizes its hyperparameters.
# It uses Decision Tree, Random Forest, and Logistic Regression classifiers, optimizing parameters like max_depth

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_transformers import data_transformer


#model_picker parameter optimizer.  Have to put this first to make the model_picker run right.  
def hyperparameter_optimizer(
        train_features: pd.DataFrame,
        train_target: pd.Series,
        valid_features: pd.DataFrame,
        valid_target: pd.Series,
        test_features: pd.DataFrame = None,
        test_target: pd.Series = None,
        model_name: str = None,
        model: str = None,
        param_name: str = None,
        model_params: dict = None,
        low: int = None,
        high: int = None,
        tolerance: float = 0.1
        ):
    
    # Initialize variables
    best_score = -np.inf
    best_param = None

    if model_params is None or model_params.get(model_name).get(param_name) is None:
        print(f'Optimizing hyperparameters:\n- Model name:{model_name}')
        print(f"-- Param {param_name}: {int(round(0))}, Accuracy Score: 0.00%")
        while high - low > tolerance:
            mid = (low + high) / 2
            
            # Set the parameter value
            params = {param_name: int(round(mid))}
            model.set_params(**params)
            
            # Fit the model        
            model.fit(train_features, train_target)
            
            # Score the model
            score = model.score(valid_features, valid_target)
            
            if score > best_score:
                best_score = score
                best_param = int(round(mid))
                low = mid
                
                # Print current param and score for debugging
                print(f"-- Param {param_name}: {int(round(mid))}, Accuracy Score: {score:.02%}")
            else:
                high = mid
        
        print(f'hyperparameter_optimizer() complete\n')
        return best_param, best_score
    else:
        # fit and score existing parameters
        model.set_params(**model_params.get(model_name))
        model.fit(train_features, train_target)
        score = model.score(test_features, test_target)
        
        return model_params, score


def best_model_picker(
        features: pd.DataFrame,
        target: pd.Series,
        test_features: pd.DataFrame = None,
        test_target: pd.Series = None,
        n_rows: int = None,
        n_target_majority: int = None,
        ordinal_cols: list = None,
        feature_scaler: bool = None,
        random_state: int = None,
        model_options: dict = None,
        model_params: dict = None,
        split_ratio: tuple = (),
        missing_values_method: str = None,
        fill_value: any = None
        ):
    # check and parameters
    print(f'Running...')
    df = pd.concat([features, target], axis=1)
    
    model_scores = {}
    optimized_hyperparameters = {}
    

    if model_options == 'all' or model_options is None:
        # models by model type so data transformation takes place once per model type. 
        model_options = {
            'Regressions': {
                'LogisticRegression': LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200)
            },
            'Machine Learning': {
                'DecisionTreeClassifier': DecisionTreeClassifier(random_state=random_state),
                'RandomForestClassifier': RandomForestClassifier(random_state=random_state),
                
            }
        }
        

    elif not (isinstance(model_options, dict) or (isinstance(model_options, str) and model_options == 'all')):
        raise ValueError("model_options must be either None or a dictionary.")
    
    
    # Loop through model types to transform data by model type
    
    for model_type, models in model_options.items():
        print(f'model_type: {model_type}')

        for model_name, model in models.items():
            print(f'model_name: {model_name}\nmodel: {model}')
        
            # Transform data
            print(f'- data_transformer()...')
            transformed_data = data_transformer(
                df=df,
                target=target.name,
                n_rows=n_rows,
                split_ratio=split_ratio,
                random_state=random_state,
                n_target_majority=n_target_majority,
                ordinal_cols=ordinal_cols,
                missing_values_method=missing_values_method,
                fill_value=fill_value,
                model_name=model_name,
                feature_scaler= feature_scaler
                )

            # Unpack transformed data
            
            # split ratio of 0 or 1 outputs a df 
            if isinstance(transformed_data, pd.DataFrame):
                print(f'transformed_data is a dataframe')
                train_features = transformed_data.drop(target.name, axis=1)
                train_target = transformed_data[target.name]
                valid_features = transformed_data.drop(target.name, axis=1)
                valid_target = transformed_data[target.name]

            else:
                print(f'transformed_data is a tuple with training data')
                train_features = transformed_data[0]
                train_target = transformed_data[1]

                if len(transformed_data) >= 4:
                    print(f'...and with validation data')
                    valid_features = transformed_data[2]
                    valid_target = transformed_data[3]
                else:
                    valid_features, valid_target = None, None

                if len(transformed_data) == 6:
                    print(f'...and with test data')
                    test_features = transformed_data[4]
                    test_target = transformed_data[5]
                # else:
                #     test_features, test_target = None, None
            

            for param_name, param_value in model.get_params().items():
                if model_name == 'RandomForestClassifier':
                    if param_name == 'max_depth':
                        print(f'- hyperparameter_optimizer(max_depth)...')
                        rfc_max_depth, rfc_best_score = hyperparameter_optimizer(
                            train_features=train_features,
                            train_target=train_target,
                            valid_features=valid_features,
                            valid_target=valid_target,
                            test_features=test_features,
                            test_target=test_target,
                            model_name=model_name,
                            model=model,
                            param_name=param_name,
                            model_params=model_params,
                            low=1,
                            high=50,
                            tolerance=0.1
                        )
                        
                        # Update optimized hyperparameters
                        optimized_hyperparameters.setdefault(model_name, {})[param_name] = rfc_max_depth

                        # Update the model with all accumulated optimized parameters
                        model.set_params(**optimized_hyperparameters[model_name])

                    elif param_name == 'n_estimators':
                        print(f'- hyperparameter_optimizer(n_estimators)...')
                        rfc_n_estimators, rfc_best_score = hyperparameter_optimizer(
                            train_features=train_features,
                            train_target=train_target,
                            valid_features=valid_features,
                            valid_target=valid_target,
                            test_features=test_features,
                            test_target=test_target,
                            model_name=model_name,
                            model=model,
                            param_name=param_name,
                            model_params=model_params,
                            low=10,
                            high=100,
                            tolerance=0.1
                        )

                        # Update optimized hyperparameters
                        optimized_hyperparameters.setdefault(model_name, {})[param_name] = rfc_n_estimators

                        # Update the model with all accumulated optimized parameters
                        model.set_params(**optimized_hyperparameters[model_name])

                        # log latest best score
                        model_scores[model_name] = rfc_best_score
                    else:

                        pass
                    
                    


                elif model_name == 'DecisionTreeClassifier':
                    if param_name == 'max_depth':
                        print(f'- hyperparameter_optimizer(max_depth)...')
                        dtc_max_depth, dtc_best_score = hyperparameter_optimizer(
                            train_features=train_features,
                            train_target=train_target,
                            valid_features=valid_features,
                            valid_target=valid_target,
                            test_features=test_features,
                            test_target=test_target,
                            model_name=model_name,
                            model=model,
                            param_name=param_name,
                            model_params=model_params,
                            low=1,
                            high=50,
                            tolerance=0.1
                        )
                        
                        # updated optimized hyperparameters log
                        optimized_hyperparameters.setdefault(model_name, {})[param_name] = dtc_max_depth

                        # Update the model with all accumulated optimized parameters
                        model.set_params(**optimized_hyperparameters[model_name])
                        
                        # log latest best score
                        model_scores[model_name] = dtc_best_score
                    else:

                        pass
                    
                    
                    

                elif model_name == 'LogisticRegression':   
                    # Fit Logistic Regression model
                    model.fit(train_features, train_target)

                    # Score the model
                    if test_features is not None and test_target is not None:
                        lr_model_score = model.score(test_features, test_target)
                    else:
                        lr_model_score = model.score(valid_features, valid_target)

                    # log latest best score
                    model_scores[model_name] = lr_model_score
                else:
                    raise ValueError(f"Unknown model: {model_name}")
    
    # Determine the best model based on validation scores
    print(f'Accuracy scores summary:\n')
    for model_name, score in model_scores.items():
        print(f'{model_name}: {score:.02%}')
        
    best_model_name = max(model_scores, key=model_scores.get)
    best_model_score = model_scores[best_model_name]
    
    print(f'best_model_picker() complete')
    return model_options, model_scores, optimized_hyperparameters, transformed_data

# reuse best model on test_df to get test results.