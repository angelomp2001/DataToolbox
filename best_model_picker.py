# This optimizes both regression model threshold and ML hyperparameters (ML auto optimizes threshold).

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_transformers import data_transformer
from metrics import categorical_scorer
import matplotlib.pyplot as plt


#model_picker parameter optimizer.  Have to put this first to make the model_picker run right.  
def hyperparameter_optimizer(
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
        threshold: float = None,
        metric: str = 'F1',
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
    print(f'running hyperparameter_optimizer()...')
    # Initialize variables
    best_score = -np.inf
    best_param = None
    score_columns=["Model Name", "Threshold", "Accuracy", "Precision", "Recall", "F1"]
    metrics = pd.DataFrame(columns=score_columns)
    model_scores = pd.DataFrame(columns=score_columns)
    


    # fit model -> get params
    model.fit(train_features, train_target)

    # designate test df
    if test_features is not None and test_target is not None:
            score_features = test_features
            score_target = test_target
    else:
        score_features = valid_features
        score_target = valid_target

    # Algo: Apply params on test df -> get y_pred
    
    print(f'hyperparameter_optimizer()/{model_type}/{model_name}')
    if model_type == 'Regressions' and model_name == 'LogisticRegression':
        print(f'if model type == {model_type}')
        
        if threshold is not None:
            #print(f'threshold is not None')
            # Score model just on this threshold
            y_pred = model.predict_proba(score_features)

            #print(f'catgorical_scorer()...')
            accuracy, precision, recall, f1 = categorical_scorer(
            target=score_target,
            y_pred=y_pred[:, 1],
            threshold=threshold
        )
            #print(f'categorical_scorer():\n{accuracy, precision, recall, f1}')
        else: # if theshold is None
            #print(f'threshold is None')
            # Score model on all thresholds <- optimization algorithms like this are the role of this function
            thresholds = np.arange(0.01, 0.99, 0.02)

            for threshold in thresholds:
                y_pred = model.predict_proba(score_features)
                # scorer is just meant to score(target, y_pred)
                #print(f'categorical_scorer()...')
                accuracy, precision, recall, f1 = categorical_scorer(
                    target=score_target,
                    y_pred=y_pred[:, 1],
                    threshold=threshold
                )
                #print(f'...catgorical_scorer: {accuracy, precision, recall, f1}')
                # log the iteration
                new_row = pd.DataFrame(
                [[model_name, threshold, accuracy, precision, recall, f1]],
                columns=score_columns
                )

                metrics = pd.concat([metrics, new_row], ignore_index=True)
                #print(f'metrics:\n{metrics.iloc[-1,:]}')
        
        # save all iterations to scores_df
        metrics_df = pd.DataFrame(metrics, columns=score_columns)

        # get max metric Score and its corresponding metrics
        best_score = metrics_df[metric].max()
        max_score_idx = metrics_df[metric].idxmax()
        best_scores = metrics_df.loc[max_score_idx]
        best_threshold = metrics_df.loc[max_score_idx, 'Threshold']
            

        #print(f'model: {model_name}\nscores:\n{best_scores}')
    
        # log best score
        print(f'best score: {best_score}')
        model_scores = pd.concat([model_scores, best_scores], ignore_index=True)

        return best_threshold, best_score, best_scores 
    
    elif model_type == 'Machine Learning':
        print(f'model type: {model_type}')
        if model_params is None or model_params.get(model_name, {}).get(param_name) is None:
            print(f'Optimizing hyperparameters:\n- Model name: {model_name}')
            print(f"- Param {param_name}: {int(round(0))}, {metric}: 0")
            
            # initialize var
            metrics = pd.DataFrame()

            # optimization algo for hyperparameters
            while high - low > tolerance:
                #print(f'-- While high-low: {high-low}')
                mid = (low + high) / 2
                
                # Set the parameter value
                params = {param_name: int(round(mid))}
                #print(f'-- params: {params}')
                model.set_params(**params)
                
                # Re/Fit the model
                model.fit(train_features, train_target)
                
                # merge categorical and hyperparameter functions: param_optimizer(hyper params, graph_scores)
                # Score the model
                y_pred = model.predict_proba(score_features)
                
                # scorer is just meant to score(target, y_pred)
                #print(f'--- categorical_scorer()...')
                accuracy, precision, recall, f1 = categorical_scorer(
                    target=score_target,
                    y_pred=y_pred[:, 1],
                    threshold=0.5 # keep constant for optimizing hyperparameters
                )
                #print(f'--- ...categorical_scorer complete.')
                # log the iteration
                new_row = pd.DataFrame(
                [[model_name, threshold, accuracy, precision, recall, f1]],
                columns=score_columns
                )

                metrics = pd.concat([metrics, new_row], ignore_index=True)
                #print(f'--- metrics:\n{metrics.iloc[-1,:]}')
                
                # Define metric for optimization
                score = new_row[metric].iloc[0]
                # print(f'--- score: {score}')
                
                if score > best_score:
                    best_score = score
                    best_param = int(round(mid))
                    low = mid + tolerance
                    
                    # Print current param and score for debugging
                    print(f"-- Param {param_name}: {int(round(mid))}, {metric}: {score:.02%}")
                else:
                    high = mid - tolerance
            
            # get metrics of best param
            metrics_df = pd.DataFrame(metrics,columns=score_columns)
            # print(f'metrics_df:\n{metrics_df.tail()}')


            # get best scores of best param
            best_scores = metrics_df.loc[metrics_df[metric] == best_score]
            # print(f'best_scores:\n{best_scores}')



            # save all iterations to scores_df
            model_scores = pd.concat([model_scores, best_scores], ignore_index=True)
            # print(f'model_scores:\n{model_scores.tail()}')

            print(f'hyperparameter_optimizer() complete\n')
            return best_param, best_score, best_scores
        
        else:
            # fit and score existing parameters
            model.set_params(**model_params.get(model_name))
            
            # No algo, just apply provided params and score
            y_pred = model.predict_proba(score_features)
            
            # scorer is just meant to score(target, y_pred)
            accuracy, precision, recall, f1 = categorical_scorer(
                target=score_target,
                y_pred=y_pred[:, 1],
                threshold=0.5 # keep constant for optimizing hyperparameters
                )

            return accuracy, precision, recall, f1
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def best_model_picker(
        features: pd.DataFrame,
        target: pd.Series,
        test_features: pd.DataFrame = None,
        test_target: pd.Series = None,
        split_ratio: tuple = (),
        n_rows: int = None,
        ordinal_cols: list = None,
        feature_scaler: bool = None,
        n_target_majority: int = None,
        missing_values_method: str = None,
        fill_value: any = None,
        random_state: int = None,
        model_options: dict = None,
        model_params: dict = None,
        metric: str = 'F1',
        graph_scores: bool = False,
        ):
    """
    1. QC parameters
    2. Data transformation by model type
    3. 
    """
    # check and parameters
    print(f'best_model_picker()...')
    df = pd.concat([features, target], axis=1)
    
    # Initialize variables
    score_columns=["Model Name", metric]
    model_scores = pd.DataFrame(columns=score_columns)
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
            ordinal_cols=ordinal_cols,
            missing_values_method=missing_values_method,
            fill_value=fill_value,
            model_type=model_type,
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

        for model_name, model in models.items():
            print(f'- for model_name: {model_name}\nmodel: {model}')
        
            # data transformer block was here

            
            
            if model_name == 'RandomForestClassifier':

                for param_name, param_value in model.get_params().items():
                    if param_name == 'max_depth':
                        print(f'-- hyperparameter_optimizer(max_depth)...')
                        rfc_max_depth, _, _ = hyperparameter_optimizer(
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
                            metric=metric,
                            graph_scores=graph_scores
                        )
                        
                        print(f'hyperparameter_optimizer() complete: max_depth: {rfc_max_depth}')
                        # Update optimized hyperparameters
                        optimized_hyperparameters.setdefault(model_name, {})[param_name] = rfc_max_depth

                        # Update the model with all accumulated optimized parameters
                        model.set_params(**optimized_hyperparameters[model_name])

                    elif param_name == 'n_estimators':
                        print(f'- hyperparameter_optimizer(n_estimators)...')
                        rfc_n_estimators, rfc_best_score, _ = hyperparameter_optimizer(
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
                            metric=metric,
                            graph_scores=graph_scores
                        )

                        print(f'hyperparameter_optimizer() complete: n_estimators: {rfc_n_estimators}')
                        # Update optimized hyperparameters
                        optimized_hyperparameters.setdefault(model_name, {})[param_name] = rfc_n_estimators

                        # Update the model with all accumulated optimized parameters
                        model.set_params(**optimized_hyperparameters[model_name])

                        # log latest best score
                        # model_scores[model_name] = rfc_best_score  # current state causes ValueError: attempt to get argmax of an empty sequence

                        new_row = pd.DataFrame(
                            [[model_name, rfc_best_score]],
                            columns=score_columns
                            )

                        model_scores = pd.concat([model_scores, new_row], ignore_index=True)

                        print(f'{model_name}:\n{model_scores}')
                    else:

                        pass

            elif model_name == 'DecisionTreeClassifier':

                for param_name, param_value in model.get_params().items():
                    if param_name == 'max_depth':
                        print(f'-- hyperparameter_optimizer(max_depth)...')
                        dtc_max_depth, dtc_best_score, _ = hyperparameter_optimizer(
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
                            metric=metric,
                            graph_scores=graph_scores
                        )
                        
                        print(f'hyperparameter_optimizer() complete: max_depth: {dtc_max_depth}')
                        # updated optimized hyperparameters log
                        optimized_hyperparameters.setdefault(model_name, {})[param_name] = dtc_max_depth

                        # Update the model with all accumulated optimized parameters
                        model.set_params(**optimized_hyperparameters[model_name])
                        
                        # log latest best score
                        new_row = pd.DataFrame(
                            [[model_name, dtc_max_depth]],
                            columns=score_columns
                            )
                        
                        model_scores = pd.concat([model_scores, new_row], ignore_index=True)

                        print(f'{model_name}:\n{model_scores}')

                    else:
                        print(f'pass')
                        pass

            elif model_name == 'LogisticRegression':
                
                lr_best_threshold, lr_best_score, lr_best_scores = hyperparameter_optimizer(
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
                            metric=metric,
                            graph_scores=graph_scores
                        )

                print(f'hyperparameter_optimizer() complete: threshold: {lr_best_threshold}')

                # log latest best score
                new_row = pd.DataFrame(
                            [[model_name, lr_best_score]],
                            columns=score_columns
                            )
                        
                model_scores = pd.concat([model_scores, new_row], ignore_index=True)
                
                print(f'{model_name}:\n{model_scores}')

            else:
                raise ValueError(f"Unknown model: {model_name}")
    
    
    # Model scores summary
    print(f'Model scores summary:\nmodel_scores:\n{model_scores}')

    # best model
    best_model_score_index = model_scores[metric].idxmax()
    best_model_scores = model_scores.loc[best_model_score_index]
 
    print(f'best model: {best_model_scores}')
    
    
    print(f'best_model_picker() complete')
    return model_scores, optimized_hyperparameters, transformed_data, model_options

# reuse best model on test_df to get test results.