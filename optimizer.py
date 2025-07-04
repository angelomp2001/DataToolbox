#model_picker parameter optimizer.  Have to put this first to make the model_picker run right.  
import pandas as pd
import numpy as np
from metrics_v2 import categorical_scorer

def optimizer(
        train_features: pd.DataFrame,
        train_target: pd.Series,
        valid_features: pd.DataFrame,
        valid_target: pd.Series,
        test_features: pd.DataFrame = None,
        test_target: pd.Series = None,
        model_type: str = None,
        model_name: str = None,
        model: str = None,
        param_to_optimize: str = None,
        model_params: dict = None,
        low: int = None,
        high: int = None,
        tolerance: float = 0.1,
        target_threshold: float = None,
        target_type: str = None,
        metric: str =  None,
        model_options: dict = None,
        ):
    """
    optimizes hyperparameter or threshold on scoring metric.
    0. designate test

    set test df
    if test df:
    - set test df

    hyperparameter optimization
    if hyperparameter:
    - if model param:
    -- if metric:
    --- optimize(model param, metric)
    ---- return optimized_param, etc.

    threshold optimization
    if classification:
    - if metric:
    -- optimize(threshold, metric)
    --- return threshold, etc.
    """

    # Initialize variables
    best_score = -np.inf
    metric_columns=["Model Name", "Threshold", "Accuracy", "Precision", "Recall", "F1"]
    metrics = pd.DataFrame(columns=metric_columns)

    "designate test df"
    if test_features is not None and test_target is not None:
            score_features = test_features
            score_target = test_target
    else:
        score_features = valid_features
        score_target = valid_target

    "hyperparameter optimization:"
    if model_type == 'Machine Learning':

        "if param_to_optimize is provided"
        if param_to_optimize is not None:
            
            #Apply any OTHER model params
            provided_model_params = model_options.get(model_type).get(model_name).get_params().copy()
            current_model_params = model.get_params().copy()
            current_model_params.update(provided_model_params)
            

            "if metric is not provided"
            if metric is None:
                metric = metric_columns[2:]
            elif not isinstance(metric, list):
                metric = [metric]

            "optimize a hyperparameter"
            # Algo for hyperparameter optimization
            for col in metric:

                while high - low > tolerance:
                    
                    # get parameter value
                    mid = (low + high) / 2
                    
                    # save the parameter value
                    params = {param_to_optimize: int(round(mid))}
                    
                    # update model params
                    current_model_params.update(params)

                    #update model with params
                    model.set_params(**current_model_params)
                    
                    #Re/Fit model
                    model.fit(train_features, train_target)
                    
                    # Predict target
                    y_pred = model.predict_proba(score_features)
                    
                    # score prediction
                    accuracy, precision, recall, f1 = categorical_scorer(
                        target=score_target,
                        y_pred=y_pred[:, 1],
                        threshold=0.5 # keep constant b/c we are optimizing hyperparameters
                    )

                    # log this iteration
                    row_data = {
                        "Model Name": model_name,
                        "Threshold": 0.5, # keep constant b/c we are optimizing hyperparameters
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1
                    }

                    row_values = pd.DataFrame([row_data])
                    metrics = pd.concat([metrics, row_values], ignore_index=True)
                    
                    # Choose metric for optimization
                    score = row_values[col].iloc[0]
                    
                    # update best score
                    if score > best_score:
                        best_score = score

                        # update model param with param of best score
                        params = {param_to_optimize: int(round(mid))}
                        

                        #set new low for next iteration
                        low = mid + tolerance
                        
                    else:
                        # set new high for next iteration
                        high = mid - tolerance            

            "Return optimized param"
            # get optimized parameter
            optimized_param = current_model_params.get(param_to_optimize)

            if metric is None or len(metric) > 1:
                pass
                # return multiple metrics
                #return optimized_param, current_model_params, metrics
            
            else:
                # get best scores of desired metric
                best_scores = metrics.loc[metrics[col] == best_score]

                # save best scores of desired metric to output
                metrics = pd.concat([metrics, best_scores], ignore_index=True)

                # return best model params, their scores, and the whole scores df
                #return optimized_param, best_score, metrics
   
        else:
            "score provided model"
            # Apply provided param to default params
            provided_model_params = model_options.get(model_type).get(model_name).get_params().copy()
            current_model_params = model.get_params().copy()
            current_model_params.update(provided_model_params)

            # set params
            model.set_params(**current_model_params)

            # fit model with params
            model.fit(train_features, train_target)
            
            # Predict Target
            y_pred = model.predict_proba(score_features)
            
            # Score Target
            accuracy, precision, recall, f1 = categorical_scorer(
                target=score_target,
                y_pred=y_pred[:, 1],
                threshold=0.5 # keep constant for optimizing hyperparameters
                )
            
            # log the results
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
            
            "Return Scores on provided parameters"
            if metric is None or len(metric) > 1:
                pass
                # return multiple metrics 
                #return current_model_params, None, metrics
            
            else:
                # get best scores of desired metric
                best_scores = metrics.loc[metrics[metric] == best_score]

                # save best scores of desired metric to output
                metrics = pd.concat([metrics, best_scores], ignore_index=True)

                #return current_model_params, best_score, metrics

    "classification optimization"
    if target_type == 'classification':

        #Apply any OTHER model params
        provided_model_params = model_options.get(model_type).get(model_name).get_params().copy()
        current_model_params = model.get_params().copy()
        current_model_params.update(provided_model_params)
        
        # Fit model
        model.fit(train_features, train_target)
        
        "if metric is not provided"
        if metric is None:
            metric = metric_columns[2:]
        elif not isinstance(metric, list):
            metric = [metric]

        "if target_threshold is not provided"        
        if target_threshold is None:
            
            "optimize target threshold"
            # initialize var
            thresholds = np.arange(0.01, 0.99, 0.02)
                   
            for threshold in thresholds:
                y_pred = model.predict_proba(score_features)
                
                # scorer is just meant to score(target, y_pred)
                accuracy, precision, recall, f1 = categorical_scorer(
                    target=score_target,
                    y_pred=y_pred[:, 1],
                    threshold=threshold
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

            "Return optimized target threshold"
            if metric is None or len(metric) > 1:
                pass
                # return multiple metrics 
                #return current_model_params, None, metrics
            
            else:
                # get max metric Score
                best_score = metrics[metric].max()

                #get the index of the max score
                max_score_idx = metrics[metric].idxmax()

                # get the row scores for max metric score 
                best_scores = metrics.loc[max_score_idx]

                # get the target threshold for max metric score
                best_target_threshold = metrics.loc[max_score_idx, 'Threshold']
                
                #return best_target_threshold, best_score, best_scores


        else:
            "if target_threshold is provided" 
            # Predict Target
            y_pred = model.predict_proba(score_features)
            
            # score prediction
            accuracy, precision, recall, f1 = categorical_scorer(
            target=score_target,
            y_pred=y_pred[:, 1],
            threshold=target_threshold
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
        
            "Return target threshold scores"
            if metric is None or len(metric) > 1:
                pass
                #return target_threshold, None, metrics
            else:
                # get max metric Score
                best_score = metrics[metric].max()

                #get the index of the max score
                max_score_idx = metrics[metric].idxmax()

                # get the row scores for max metric score 
                all_target_threshold_scores = metrics.loc[max_score_idx]
                
                #return target_threshold, best_score, all_target_threshold_scores
            
    if model_type == 'Machine Learning':
        if param_to_optimize is not None:
            if metric is None or len(metric) > 1:
                return optimized_param, current_model_params, metrics #(150) ✅
            else:
                return optimized_param, best_score, metrics #(160) ✅
        else:
            if metric is None or len(metric) > 1:
                # return multiple metrics 
                return current_model_params, None, metrics #(207) ✅
            else:
                 return current_model_params, best_score, metrics #(216) ✅
    
    if target_type == 'classification':
        if target_threshold is None:
            if metric is None or len(metric) > 1:
                # return multiple metrics 
                return current_model_params, None, metrics #(269) ✅
            else:
                return best_target_threshold, best_score, best_scores #(284) ✅
        else:
            if metric is None or len(metric) > 1:
                return target_threshold, None, metrics #(313) ✅
            else:
                return target_threshold, best_score, all_target_threshold_scores #(325) ✅