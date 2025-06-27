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
        threshold: float = 0.5,
        metric: str = 'f1',
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
    if model_type == 'Regressions' and model_name == 'LogisticRegression':
        if threshold is not None:
            # Score model just on this threshold
            y_pred = model.predict_proba(score_features)

            accuracy, precision, recall, f1 = categorical_scorer(
            target=score_target,
            y_pred=y_pred[:, 1],
            graph_scores=True,
            model_type=model_type,
            model_name=model_name,
            threshold=threshold
        )
        else: # if theshold is None
            # Score model on all thresholds <- optimization algorithms like this are the role of this function
            thresholds = np.arange(0.01, 0.99, 0.02)
            metrics = []

            for threshold in thresholds:
                y_pred = model.predict_proba(score_features)
                # scorer is just meant to score(target, y_pred)
                accuracy, precision, recall, f1 = categorical_scorer(
                    target=score_target,
                    y_pred=y_pred[:, 1],
                    graph_scores=True,
                    model_type=model_type,
                    model_name=model_name,
                    threshold=threshold
                )

                # log the iteration
                metrics.append([model_name,threshold, accuracy, precision, recall, f1])
        
            # save all iterations to scores_df
            scores_df = pd.DataFrame(metrics, columns=['Model Name','Threshold', 'Accuracy', 'Precision', 'Recall', 'F1'])

            # get max F1 Score and its corresponding metrics
            max_f1_idx = scores_df['F1'].idxmax()
            best_scores = scores_df.loc[max_f1_idx]

            print(f'model: {model_name}\nscores:\n{best_scores}')
    
        # log best score
        model_scores = pd.concat([model_scores, best_scores], ignore_index=True)

        return model_scores
    

    elif model_type == 'Machine Learning':
        if model_params is None or model_params.get(model_name).get(param_name) is None:
            print(f'Optimizing hyperparameters:\n- Model name:{model_name}')
            print(f"-- Param {param_name}: {int(round(0))}, Accuracy Score: 0.00%")
            
            # optimization algo for hyperparameters
            while high - low > tolerance:
                mid = (low + high) / 2
                
                # Set the parameter value
                params = {param_name: int(round(mid))}
                model.set_params(**params)
                
                # merge categorical and hyperparameter functions: param_optimizer(hyper params, graph_scores)
                # Score the model
                y_pred = model.predict_proba(score_features)
                
                # scorer is just meant to score(target, y_pred)
                accuracy, precision, recall, f1 = categorical_scorer(
                    target=score_target,
                    y_pred=y_pred[:, 1],
                    graph_scores=True,
                    model_name=model_name,
                    model_type=model_type,
                )
                
                # log the iteration
                metrics.append([model_name, threshold, accuracy, precision, recall, f1])
                
                # Define metric for optimization
                score = metrics[metric]
                
                if score > best_score:
                    best_score = score
                    best_param = int(round(mid))
                    low = mid
                    
                    # Print current param and score for debugging
                    print(f"-- Param {param_name}: {int(round(mid))}, Accuracy Score: {score:.02%}")
                else:
                    high = mid
                        
            # save all iterations to scores_df
            model_scores = pd.concat([model_scores, best_scores], ignore_index=True)
            
            print(f'hyperparameter_optimizer() complete\n')
            return best_param, best_score
        
        else:
            # fit and score existing parameters
            model.set_params(**model_params.get(model_name))
            
            # No algo, just apply provided params and score
            y_pred = model.predict_proba(score_features)
            
            # scorer is just meant to score(target, y_pred)
            accuracy, precision, recall, f1 = categorical_scorer(
                target=score_target,
                y_pred=y_pred[:, 1],
                graph_scores=True
                )
            
            

    if graph_scores:
        model_name, accuracy, precision, recall, f1 = model_scores
        plt.figure(figsize=(8, 6))
        plt.plot(model_scores['Recall'], model_scores['Precision'], label="Precision-Recall Curve")
        plt.scatter(best_recall, best_precision,
                    color='red',
                    label=f'Max F1 = {max_f1:.2f}\nThreshold = {best_threshold:.2f}',
                    zorder=5)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve with Max F1 Indication")
        plt.legend()
        plt.grid(True)
        plt.show()

    return accuracy, precision, recall, f1

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
        metric: str = 'f1',
        graph_scores: bool = False,
        ):
    # check and parameters
    print(f'Running...')
    df = pd.concat([features, target], axis=1)
    
    # Initialize variables
    model_scores = pd.DataFrame()
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
            
            if model_name == 'RandomForestClassifier':
                for param_name, param_value in model.get_params().items():
            # for param_name, param_value in model.get_params().items():
            #     if model_name == 'RandomForestClassifier':
                    if param_name == 'max_depth':
                        print(f'- hyperparameter_optimizer(max_depth)...')
                        rfc_max_depth, _ = hyperparameter_optimizer(
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
                            tolerance=0.1,
                            metric=metric,
                            graph_scores=graph_scores
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
                            tolerance=0.1,
                            metric=metric,
                            graph_scores=graph_scores
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
                for param_name, param_value in model.get_params().items():
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
                            tolerance=0.1,
                            metric=metric,
                            graph_scores=graph_scores
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
                model_scores = hyperparameter_optimizer(
                            train_features=train_features,
                            train_target=train_target,
                            valid_features=valid_features,
                            valid_target=valid_target,
                            test_features=test_features,
                            test_target=test_target,
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

            else:
                raise ValueError(f"Unknown model: {model_name}")
    
    
    # Model scores summary
    print(f'Model scores summary:\nmodel_scores:\n{model_scores}')

    # best model
    best_model_score_index = model_scores['F1 Score'].idxmax()
    max_f1_row = model_scores.loc[best_model_score_index]
 
    print(f'best model: {max_f1_row}')
    
    
    print(f'best_model_picker() complete')
    return model_scores, optimized_hyperparameters, transformed_data, model_options

# reuse best model on test_df to get test results.