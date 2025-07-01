# This optimizes target_threshold and ML hyperparameters.

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_transformers_v2 import data_transformer
from metrics_v2 import categorical_scorer
import matplotlib.pyplot as plt


#model_picker parameter optimizer.  Have to put this first to make the model_picker run right.  
def model_score_optimizer(
        train_features: pd.DataFrame,
        train_target: pd.Series,
        valid_features: pd.DataFrame,
        valid_target: pd.Series,
        test_features: pd.DataFrame = None,
        test_target: pd.Series = None,
        model_type: str = None,
        model_name: str = None,
        model: str = None,
        param_name: str = None,
        model_params: dict = None,
        low: int = None,
        high: int = None,
        tolerance: float = 0.1,
        target_threshold: float = None,
        target_type: str = None,
        metric: str =  None,
        graph_scores: bool = False,
        ):
    """
    parameter optimization algos feed into scoring function.
    1. fit model -> get params
    2. designate test df
    3. Algo: Apply params on test df -> get y_pred
        score parameters(target, y_pred)
    4. print results
    """
    # Initialize variables
    best_score = -np.inf
    best_param = None
    metric_columns=["Model Name", "Threshold", "Accuracy", "Precision", "Recall", "F1"]
    model_scores = pd.DataFrame(columns=metric_columns)

    # designate test df
    if test_features is not None and test_target is not None:
            score_features = test_features
            score_target = test_target
    else:
        score_features = valid_features
        score_target = valid_target

    # Algo: optimize hyperparameters for ML models
    if model_type == 'Machine Learning':

        # only apply if hyperparameters is not provided
        if (model_params is None or model_params.get(model_name, {}).get(param_name) is None) or (model_params is not None and model_params.get(model_name, {}).get(param_name) is None):
            
            # initialize var
            metrics = pd.DataFrame()

            if metric is None:
                metric = metric_columns[2:]
            elif not isinstance(metric, list):
                metric = [metric]

            
            for col in metric:
                # optimization algo for hyperparameters
                while high - low > tolerance:
                    #print(f'-- While high-low: {high-low}')
                    mid = (low + high) / 2
                    
                    # Set the parameter value
                    params = {param_name: int(round(mid))}
                    #print(f'-- params: {params}')
                    model.set_params(**params)
                    
                    # Re/Fit the model when hyperparams exist
                    model.fit(train_features, train_target)
                    
                    # merge categorical and hyperparameter functions: param_optimizer(hyper params, graph_scores)
                    # Score the model
                    y_pred = model.predict_proba(score_features)
                    
                    # scorer is just meant to score(target, y_pred)
                    accuracy, precision, recall, f1 = categorical_scorer(
                        target=score_target,
                        y_pred=y_pred[:, 1],
                        target_threshold=0.5 # keep constant for optimizing hyperparameters
                    )

                    # log the iteration
                    row_data = {
                        "Model Name": model_name,
                        "Threshold": target_threshold,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1
                    }

                    row_values = pd.DataFrame([row_data])
                    metrics = pd.concat([metrics, row_values], ignore_index=True)
                    
                    # Choose metric for optimization
                    score = row_values[col].iloc[0]
                    
                    if score > best_score:
                        best_score = score
                        model_params = int(round(mid))
                        low = mid + tolerance
                        
                    else:
                        high = mid - tolerance
                
                # get metrics of best param
                metrics_df = pd.DataFrame(metrics,columns=metric_columns)
            
                print(f'{model_params=}')
                
############################################################ return  starts here #####################################
                if metric is None or len(metric) > 1:
                    return model_params, None, metrics_df
                else:

                    # get best scores of best param
                    best_scores = metrics_df.loc[metrics_df[col] == best_score]

                    # save all iterations to scores_df
                    model_scores = pd.concat([model_scores, best_scores], ignore_index=True)
                    # print(f'model_scores:\n{model_scores.tail()}')

                    print(f'model_score_optimizer() complete\n')
                    return model_params, best_score, model_scores
#########################################################################################      
        else:
            # fit and score existing parameters
            print(f'setting provided params: {model_params.get(model_name)}')
            model.set_params(**model_params.get(model_name))
            model.fit(train_features, train_target)
            
            # No algo, just apply provided params and score
            y_pred = model.predict_proba(score_features)
            
            # scorer is just meant to score(target, y_pred)
            accuracy, precision, recall, f1 = categorical_scorer(
                target=score_target,
                y_pred=y_pred[:, 1],
                target_threshold=0.5 # keep constant for optimizing hyperparameters
                )
            
            # log the iteration
            row_data = {
                "Model Name": model_name,
                "Threshold": threshold,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            }

            row_values = pd.DataFrame([row_data])
            metrics = pd.concat([metrics, row_values], ignore_index=True)

            # get metrics of best param
            metrics_df = pd.DataFrame(metrics,columns=metric_columns)
            
            # get best score
            best_score = metrics_df[metric].max()

            # get best scores of best param
            best_scores = metrics_df.loc[metrics_df[metric] == best_score]
            
######################### return starts here ###################################
            

            if metric is None or len(metric) > 1:
                return None, None, metrics_df
            else:
                print(f'hyperparameter_optimizer() complete\n')
                return model_params, best_score, best_scores
###########################################################################

    # Algo: optimize threshold when target_classification
    if target_type == 'classification':
        
        # ML would be fit by now, but not regressions
        if model_type != 'Machine Learning':
            model.fit(train_features, train_target)
        else:
            pass

        if target_threshold is not None:
            #print(f'target_threshold is not None')
            # Score model just on this target_threshold
            y_pred = model.predict_proba(score_features)

            # scorer is just meant to score(target, y_pred)
            accuracy, precision, recall, f1 = categorical_scorer(
            target=score_target,
            y_pred=y_pred[:, 1],
            target_threshold=target_threshold
            )

            # log the iteration
            row_data = {
                "Model Name": model_name,
                "Threshold": target_threshold,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            }

            

            row_values = pd.DataFrame([row_data])
            model_scores = pd.concat([model_scores, row_values], ignore_index=True)

        else: # if theshold is None
            # Score model on all thresholds <- optimization algorithms like this are the role of this function
            thresholds = np.arange(0.01, 0.99, 0.02)

            for threshold in thresholds:
                y_pred = model.predict_proba(score_features)
                
                # scorer is just meant to score(target, y_pred)
                accuracy, precision, recall, f1 = categorical_scorer(
                    target=score_target,
                    y_pred=y_pred[:, 1],
                    target_threshold=threshold
                )

                # log the iteration
                row_data = {
                    "Model Name": model_name,
                    "Threshold": threshold,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1
                }

                row_values = pd.DataFrame([row_data])
                model_scores = pd.concat([model_scores, row_values], ignore_index=True)
        
        # save all iterations to scores_df
        # metrics_df = pd.DataFrame(model_scores, columns=metric_columns)
##########################################################################################
        if metric is None or len(metric) > 1:
            return None, None, model_scores
        else:
            # get max metric Score and its corresponding metrics
            best_score = model_scores[metric].max()
            max_score_idx = model_scores[metric].idxmax()
            best_scores = model_scores.loc[max_score_idx]
            best_target_threshold = model_scores.loc[max_score_idx, 'Threshold']
        
            # log best score
            model_scores = pd.concat([model_scores, best_scores], ignore_index=True)
            
            print(f'model_score_optimizer() complete\n')
            return best_target_threshold, best_score, best_scores            
#############################################################################################        
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
    1. QC parameters
    2. Data transformation by model type
    3. 
    """
    # check and parameters
    df = pd.concat([features, target], axis=1)
    
    # Initialize variables
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
    
    
    # Data Transformation: Loop through model types to transform data by model type
    
    for model_type, models in model_options.items():
        print(f'for model_type: {model_type}')

        # Transform data
        print(f'- data_transformer()...')
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
            print(f'transformed_data is a dataframe')
            train_features = transformed_data.drop(target.name, axis=1)
            train_target = transformed_data[target.name]
            valid_features = transformed_data.drop(target.name, axis=1)
            valid_target = transformed_data[target.name]

            print(f'df count: {len(train_features)}')

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

        for model_name, model in models.items():
            
            if model_name == 'RandomForestClassifier':
                if model_params is not None and model_name in model_params:
                    params_to_iterate = model_params[model_name].items()
                    print(f'provided params to iterate: {params_to_iterate}')
                else:
                    params_to_iterate = model.get_params().items()
  
                for param_name, param_value in params_to_iterate:        
                    if param_name == 'max_depth':
                        rfc_max_depth, _, _ = model_score_optimizer(
                            train_features=train_features,
                            train_target=train_target,
                            valid_features=valid_features,
                            valid_target=valid_target,
                            test_features=test_features,
                            test_target=test_target,
                            model_type=model_type,
                            model_name=model_name,
                            model=model,
                            param_name=param_name,
                            model_params=model_params,
                            low=1,
                            high=50,
                            tolerance=0.1,
                            target_threshold=target_threshold,
                            metric=metric,
                            target_type = target_type,
                            graph_scores=graph_scores
                        )
                        print(f'{rfc_max_depth=}')
                        # Update optimized hyperparameters
                        optimized_hyperparameters.setdefault(model_name, {})[param_name] = rfc_max_depth

                        # Update the model with all accumulated optimized parameters
                        model.set_params(**optimized_hyperparameters[model_name])

                    elif param_name == 'n_estimators':
                        print(f'-- hyperparameter_optimizer(n_estimators)...')
                        rfc_n_estimators, _, rfc_scores = model_score_optimizer(
                            train_features=train_features,
                            train_target=train_target,
                            valid_features=valid_features,
                            valid_target=valid_target,
                            test_features=test_features,
                            test_target=test_target,
                            model_type=model_type,
                            model_name=model_name,
                            model=model,
                            param_name=param_name,
                            model_params=model_params,
                            low=10,
                            high=100,
                            tolerance=0.1,
                            target_threshold=target_threshold,
                            metric=metric,
                            target_type = target_type,
                            graph_scores=graph_scores
                        )
                        print(f'{rfc_n_estimators=}')
                        # Update optimized hyperparameters
                        optimized_hyperparameters.setdefault(model_name, {})[param_name] = rfc_n_estimators

                        # Update the model with all accumulated optimized parameters
                        model.set_params(**optimized_hyperparameters[model_name])

                        # log the iteration
                        row_values = rfc_scores.copy()
                        model_scores = pd.concat([model_scores, row_values], ignore_index=True)

                    else:

                        pass

            elif model_name == 'DecisionTreeClassifier':

                for param_name, param_value in model.get_params().items():
                    if param_name == 'max_depth':
                        
                        dtc_max_depth, _, dtc_best_scores = model_score_optimizer(
                            train_features=train_features,
                            train_target=train_target,
                            valid_features=valid_features,
                            valid_target=valid_target,
                            test_features=test_features,
                            test_target=test_target,
                            model_type=model_type,
                            model_name=model_name,
                            model=model,
                            param_name=param_name,
                            model_params=model_params,
                            low=1,
                            high=50,
                            tolerance=0.1,
                            target_threshold=target_threshold,
                            metric=metric,
                            target_type = target_type,
                            graph_scores=graph_scores
                        )
                        print(f'{dtc_max_depth=}')
                        # updated optimized hyperparameters log
                        optimized_hyperparameters.setdefault(model_name, {})[param_name] = dtc_max_depth

                        # Update the model with all accumulated optimized parameters
                        model.set_params(**optimized_hyperparameters[model_name])
                        
                        # log the iteration
                        row_values = dtc_best_scores.copy()
                        model_scores = pd.concat([model_scores, row_values], ignore_index=True)

                    else:

                        pass

            elif model_name == 'LogisticRegression':
                
                lr_best_target_threshold, lr_best_score, lr_best_scores = model_score_optimizer(
                            train_features=train_features,
                            train_target=train_target,
                            valid_features=valid_features,
                            valid_target=valid_target,
                            test_features=test_features,
                            test_target=test_target,
                            model_type=model_type,
                            model_name=model_name,
                            model=model,
                            param_name= None,
                            model_params=model_params,
                            low=1,
                            high=50,
                            tolerance=0.1,
                            target_threshold=target_threshold,
                            metric=metric,
                            target_type = target_type,
                            graph_scores=graph_scores
                        )

                # log the iteration
                row_values = lr_best_scores.copy()
                model_scores = pd.concat([model_scores, row_values], ignore_index=True)

            else:
                raise ValueError(f"Unknown model: {model_name}")
    
##############################################################################################    
    # Model scores summary
    #print(f'Model scores summary:\nmodel_scores:\n{model_scores}')


    if metric is None or len(metric) > 1:
        # Create a summary table
        best_model_scores = []

        for score_name in model_scores.columns[2:]:
            best_score_index = model_scores[score_name].idxmax()
            best_score = model_scores.loc[best_score_index, score_name]
            model_name = model_scores.loc[best_score_index, 'Model Name']
            best_model_scores.append({
                'Metric Name': score_name,
                'Best Score': best_score,
                'Model Name': model_name
            })

        best_scores_summary_df = pd.DataFrame(best_model_scores)
        print(best_scores_summary_df)

        # the optimizied parameters is empty
        return best_scores_summary_df, optimized_hyperparameters, transformed_data, model_options

    else:

        # best model
        best_model_score_index = model_scores[metric].idxmax()
        best_model_scores = model_scores.loc[best_model_score_index]
    
        print(f'best model scores:\n{best_model_scores}')
        print(f'best_model_picker() complete')
        return best_model_scores, optimized_hyperparameters, transformed_data, model_options

# reuse best model on test_df to get test results.   
