# This optimizes target_threshold and ML hyperparameters.

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_transformers_v3 import data_transformer
from metrics_v2 import categorical_scorer
import matplotlib.pyplot as plt
from optimizer import optimizer
from sklearn.metrics import roc_auc_score, average_precision_score


def best_model_picker(
        features: pd.DataFrame,
        target: pd.Series,
        test_features: pd.DataFrame = None,
        test_target: pd.Series = None,
        split_ratio: tuple = (),
        missing_values_method: str = None,
        fill_value: any = None,
        n_rows: int = None,
        n_target_majority: int = None,
        n_target_minority: int = None,
        random_state: int = None,
        scale_features: bool = None,
        ordinal_cols: list = None,
        target_type: str = None,
        model_options: dict = None,
        model_params: dict = None,
        target_threshold: float = None,
        metric: str = None,
        graph_scores: bool = False,
        ):
    """
    0. Initialize vars
    1. Data transformation by model type
        - Split data
        -- Iterate through models and their hyperparameters
        --- Optimize certain hyperparameters
    2. Report findings
        - Summary by score(s)
        -- Summary by model
    """
    print(f'Running model picker...')
    
    # Initialize variables
    df = pd.concat([features, target], axis=1)
    model_scores = pd.DataFrame()
    optimized_hyperparameters = {}
    
    # Ensure model_options is not None
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
    
    
    "Data Transformation by model type"
    for model_type, models in model_options.items():
        print(f'model type: {model_type}')
        
        # Transform data
        

        transformed_data = data_transformer(
            df=df,
            target=target.name,
            n_rows=n_rows,
            split_ratio=split_ratio,
            random_state=random_state,
            n_target_majority=n_target_majority,
            n_target_minority=n_target_minority,
            ordinal_cols=ordinal_cols,
            missing_values_method=missing_values_method,
            fill_value=fill_value,
            model_type=model_type,
            scale_features= scale_features
            )
        
        # Unpack transformed data
        # split ratio of 0 or 1 outputs a df 
        if isinstance(transformed_data, pd.DataFrame):
            
            train_features = transformed_data.drop(target.name, axis=1)
            train_target = transformed_data[target.name]
            valid_features = transformed_data.drop(target.name, axis=1)
            valid_target = transformed_data[target.name]

            

        else:
            train_features = transformed_data[0]
            train_target = transformed_data[1]

            if len(transformed_data) >= 4:
                valid_features = transformed_data[2]
                valid_target = transformed_data[3]
            else:
                valid_features, valid_target = None, None

            if len(transformed_data) == 6:
                test_features = transformed_data[4]
                test_target = transformed_data[5]
            else:
               test_features, test_target = None, None

        "Iterate through models and their hyperparameters"
        for model_name, model in models.items():
            print(f'model name: {model_name}')
            if model_name == 'RandomForestClassifier':

                #get model_name params
                model_name_hyperparams = (model_params or {}).get(model_name)
                
                #set params for iteration
                if model_name_hyperparams is not None:
                    params_to_iterate = model_name_hyperparams.items()
                else:
                    params_to_iterate = model.get_params().items()

                # iterate through params
                for param_to_optimize, param_value in params_to_iterate:        
                    if param_to_optimize == 'max_depth':
                        print(f'param_to_optimize: {param_to_optimize}')
                        
                        
                        rfc_max_depth, _, _ = optimizer(
                            train_features=train_features,
                            train_target=train_target,
                            valid_features=valid_features,
                            valid_target=valid_target,
                            test_features=test_features,
                            test_target=test_target,
                            model_type=model_type,
                            model_name=model_name,
                            model=model,
                            param_to_optimize=param_to_optimize,
                            model_params=model_params,
                            low=1,
                            high=100, # was 50
                            tolerance=0.1,
                            target_threshold=target_threshold,
                            metric=metric,
                            model_options=model_options,
                            target_type=target_type
                        )
                        
                        # Update optimized hyperparameters
                        optimized_hyperparameters.setdefault(model_name, {})[param_to_optimize] = rfc_max_depth

                        # Update the model with all accumulated optimized parameters
                        model.set_params(**optimized_hyperparameters[model_name])

                    elif param_to_optimize == 'n_estimators':
                        print(f'-- hyperparameter_optimizer(n_estimators)...')
                        rfc_n_estimators, _, rfc_scores = optimizer(
                            train_features=train_features,
                            train_target=train_target,
                            valid_features=valid_features,
                            valid_target=valid_target,
                            test_features=test_features,
                            test_target=test_target,
                            model_type=model_type,
                            model_name=model_name,
                            model=model,
                            param_to_optimize=param_to_optimize,
                            model_params=model_params,
                            low=10,
                            high=100,
                            tolerance=0.1,
                            target_threshold=target_threshold,
                            metric=metric,
                            model_options=model_options,
                            target_type=target_type
                        )
                        
                        # Update optimized hyperparameters
                        optimized_hyperparameters.setdefault(model_name, {})[param_to_optimize] = rfc_n_estimators

                        # Update the model with all accumulated optimized parameters
                        model.set_params(**optimized_hyperparameters[model_name])
                        

                        # log the iteration
                        row_values = rfc_scores.copy()
                        model_scores = pd.concat([model_scores, row_values], ignore_index=True)

                    else:

                        pass

            elif model_name == 'DecisionTreeClassifier':
                
                for param_to_optimize, param_value in model.get_params().items():
                    if param_to_optimize == 'max_depth':
                        
                        dtc_max_depth, _, dtc_best_scores = optimizer(
                            train_features=train_features,
                            train_target=train_target,
                            valid_features=valid_features,
                            valid_target=valid_target,
                            test_features=test_features,
                            test_target=test_target,
                            model_type=model_type,
                            model_name=model_name,
                            model=model,
                            param_to_optimize=param_to_optimize,
                            model_params=model_params,
                            low=1,
                            high=50,
                            tolerance=0.1,
                            target_threshold=target_threshold,
                            metric=metric,
                            model_options=model_options,
                            target_type=target_type
                        )
                        
                        # updated optimized hyperparameters log
                        optimized_hyperparameters.setdefault(model_name, {})[param_to_optimize] = dtc_max_depth

                        # Update the model with all accumulated optimized parameters
                        model.set_params(**optimized_hyperparameters[model_name])
                        
                        # log the iteration
                        row_values = dtc_best_scores.copy()
                        model_scores = pd.concat([model_scores, row_values], ignore_index=True)

                    else:

                        pass

            elif model_name == 'LogisticRegression':
                
                lr_best_target_threshold, lr_best_score, lr_best_scores = optimizer(
                            train_features=train_features,
                            train_target=train_target,
                            valid_features=valid_features,
                            valid_target=valid_target,
                            test_features=test_features,
                            test_target=test_target,
                            model_type=model_type,
                            model_name=model_name,
                            model=model,
                            param_to_optimize= None,
                            model_params=model_params,
                            low=1,
                            high=50,
                            tolerance=0.1,
                            metric=metric,
                            target_type = target_type,
                            model_options=model_options,
                            target_threshold=target_threshold
                        )

                # log the iteration
                row_values = lr_best_scores.copy()
                model_scores = pd.concat([model_scores, row_values], ignore_index=True)

            else:
                raise ValueError(f"Unknown model: {model_name}")
    
  
    "Report summary scores"
    if metric is None or len(metric) > 1:
        # Create a summary table by score
        best_model_scores = []

        "Summary by score"
        for score_name in model_scores.columns[2:-2]:
            numeric_score_series = pd.to_numeric(model_scores[score_name], errors='coerce') 
            best_score_index = numeric_score_series.idxmax()
            # best_score_index = model_scores[score_name].idxmax()
            best_score = model_scores.loc[best_score_index, score_name]
            model_name = model_scores.loc[best_score_index, 'Model Name']
            best_model_scores.append({
                'Metric Name': score_name,
                #'Parameter': model_scores.loc[best_score_index, 'Parameter'],
                'Best Score': best_score,
                'Model Name': model_name
            })

        # print best scores summary table
        best_scores_summary_df = pd.DataFrame(best_model_scores)
        print(f'best_scores_summary_df:\n{best_scores_summary_df}')

        "summary by model"
        best_scores_by_model = []

        for each_model in model_scores['Model Name'].unique():
            for score_name in model_scores.columns[2:-2]:  # Assuming columns[2:] are metric columns
                # Get best score
                best_score = model_scores[model_scores['Model Name'] == each_model][score_name].max()

                # get best score corresponding threshold
                best_score_index = model_scores[model_scores['Model Name'] == each_model][score_name].idxmax()
                best_threshold = model_scores.loc[best_score_index, 'Threshold']
                best_roc_auc = model_scores.loc[best_score_index, 'ROC AUC']
                best_pr_auc = model_scores.loc[best_score_index, 'PR AUC']
                
                # Append to the results table
                best_scores_by_model.append({
                    'Model Name': each_model,
                    #'Parameter': best_threshold,
                    'Metric': score_name,
                    'Score': best_score,
                    'ROC AUC': best_roc_auc,
                    'PR AUC': best_pr_auc
                })

        best_scores_by_model_df = pd.DataFrame(best_scores_by_model)
        # print(f'best_scores_by_model_df:\n{best_scores_by_model_df}')
        # pd.set_option('display.max_rows', None)        # Display all rows
        # pd.set_option('display.max_columns', None)     # Display all columns
        # pd.set_option('display.width', 1000)           # Make sure the console is wide enough to avoid wrapping
        # pd.set_option('display.colheader_justify', 'left') # Align column headers for better readability
        #print(f'model_scores:\n{model_scores}')
        # pd.reset_option('display.max_rows')
        # pd.reset_option('display.max_columns')
        # pd.reset_option('display.width')
        # pd.reset_option('display.colheader_justify')

        
        print(f'optimized_hyperparameters: {optimized_hyperparameters}')
        return best_scores_summary_df, optimized_hyperparameters, best_scores_by_model_df, model_scores, transformed_data, model_options

    else:

        # best model
        best_model_score_index = model_scores[metric].idxmax()
        best_model_scores = model_scores.loc[best_model_score_index]
    
        print(f'best model scores:\n{best_model_scores}')
        return best_model_scores, optimized_hyperparameters, None, model_scores, transformed_data, model_options

# reuse best model on test_df to get test results.   
