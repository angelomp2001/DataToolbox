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

#model_picker parameter optimizer.  Have to put this first to make the model_picker run right.  
def hyperparameter_optimizer(model, param_name, low, high, train_features, train_target, valid_features, valid_target, model_name, random_state, tolerance=0.1):
    best_score = -np.inf
    best_param = None
    print(f'Optimizing hyperparameters\nModel name:{model_name}\n')

    #merge df and encode categorical vars if applicable
    print(f'merging data to encode categorical vars')
    df_train = pd.concat([train_features, train_target], axis = 1)
    df_valid = pd.concat([valid_features, valid_target], axis= 1)
    df = pd.concat([df_train, df_valid], axis = 0)
    df = df.dropna()
    # encoder = OrdinalEncoder()
    # encoder.fit(df)
    # data_ordinal = encoder.transform(df)
    # data_ordinal = pd.DataFrame(encoder.transform(df), columns=df.columns)

    # #re-split df
    # print(f're-splitting data...')
    # df_train, df_other = train_test_split(data_ordinal, test_size=0.4, random_state=random_state)  # training is 60%
    # df_valid, df_test = train_test_split(df_other, test_size=0.5, random_state=random_state)  # valid & test = .4 * .5 = .2 each
    # print(f'\
    # df_train:{df_train.shape}\n\
    # df_valid: {df_valid.shape}\n\
    # df_test:{df_test.shape}')
    
    # # Define features and targets per df
    # print(f'defining features and targets among split data before hyperparameter optimization')
    # train_features = df_train.drop(target.name, axis=1)
    # train_target = df_train[target.name]
    # valid_features = df_valid.drop(target.name, axis=1)
    # valid_target = df_valid[target.name]
    
    
    while high - low > tolerance:
        mid = (low + high) / 2
        
        # Set the parameter value
        params = {param_name: int(round(mid))}
        model.set_params(**params)
        
        # Fit the model        
        model.fit(train_features, train_target)
        
        # Score the model
        score = model.score(valid_features, valid_target)
        
        # Print current param and score for debugging
        print(f"Param {param_name}: {int(round(mid))}, Accuracy Score: {score:.02%}")
        
        if score > best_score:
            best_score = score
            best_param = int(round(mid))
            low = mid
        else:
            high = mid
    
    return best_param, best_score

def model_picker(features, target, ordinal_cols=None):
    #merge data
    random_state = 12345
    df = pd.concat([features, target], axis=1)  # Ensure features and target are combined into a single DataFrame
    df = df.dropna()

    # # split data
    # print(f'splitting data...')
    # df_train, df_other = train_test_split(df, test_size=0.4, random_state=random_state)  # training is 60%
    # df_valid, df_test = train_test_split(df_other, test_size=0.5, random_state=random_state)  # valid & test = .4 * .5 = .2 each
    # print(f'split data:\n\
    # df_train:{df_train.shape}\n\
    # df_valid: {df_valid.shape}\n\
    # df_test:{df_test.shape}')
    
    # # Define features and targets per df
    # print(f'defining features and targets among split data')
    # train_features = df_train.drop(target.name, axis=1)
    # train_target = df_train[target.name]
    # valid_features = df_valid.drop(target.name, axis=1)
    # valid_target = df_valid[target.name]
    # test_features = df_test.drop(target.name, axis=1)
    # test_target = df_test[target.name]
    
    # Define base models
    dtc_model = DecisionTreeClassifier(random_state=random_state)
    rfc_model = RandomForestClassifier(random_state=random_state)
    lr_model = LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200)
    
    
    # Optimize max_depth for DecisionTreeClassifier
    dtc_max_depth, dtc_best_score = hyperparameter_optimizer(
        dtc_model, 'max_depth', 1, 20,
        train_features=train_features, train_target=train_target,
        valid_features=valid_features, valid_target=valid_target, model_name = "DecisionTreeClassifier", random_state=random_state
    )
        
    # Optimize max_depth and n_estimators for RandomForestClassifier
    rfc_max_depth, _ = hyperparameter_optimizer(
        rfc_model, 'max_depth', 1, 20,
        train_features=train_features, train_target=train_target,
        valid_features=valid_features, valid_target=valid_target, model_name = "RandomForestClassifier", random_state=random_state
    )
    rfc_n_estimators, rfc_best_score = hyperparameter_optimizer(
        rfc_model, 'n_estimators', 10, 100,
        train_features=train_features, train_target=train_target,
        valid_features=valid_features, valid_target=valid_target, model_name = "RandomForestClassifier", random_state=random_state
    )

    rfc_params = {'max_depth': rfc_max_depth, 'n_estimators': rfc_n_estimators}
    rfc_model.set_params(**rfc_params)
    
    # rfc_model.set_params(max_depth=rfc_max_depth, n_estimators=rfc_n_estimators)

    # # merge df, encode, split df
    # data_ohe = pd.get_dummies(df.dropna(), drop_first=True)
    # df_train, df_other = train_test_split(data_ohe, test_size=0.4, random_state=random_state)  # training is 60%
    # df_valid, df_test = train_test_split(df_other, test_size=0.5, random_state=random_state)  # valid & test = .4 * .5 = .2 each
    # print(f'one-hot encoding + split data with logistic regression:\n\
    # df_train:{df_train.shape}\n\
    # df_valid: {df_valid.shape}\n\
    # df_test:{df_test.shape}')
    
    # # Define features and targets per df
    # train_features = df_train.drop(target.name, axis=1)
    # train_target = df_train[target.name]
    # valid_features = df_valid.drop(target.name, axis=1)
    # valid_target = df_valid[target.name]
    # test_features = df_test.drop(target.name, axis=1)
    # test_target = df_test[target.name]
    

    # Fit Logistic Regression model
    lr_model.fit(train_features, train_target)
    lr_model_score = lr_model.score(valid_features, valid_target)
    
    # Determine the best model based on validation scores
    best_scores = {
        'DecisionTreeClassifier': dtc_best_score,
        'RandomForestClassifier': rfc_best_score,
        'LogisticRegression': lr_model_score
    }
    best_model_name = max(best_scores, key=best_scores.get)
    
    print(f'Accuracy scores summary:\n\
          DecisionTreeClassifier: {dtc_best_score},\n\
          RandomForestClassifier: {rfc_best_score},\n\
          LogisticRegression: {lr_model_score}\
            ')

    #commenting out to test all models on test data to see which is really the best
    # # Retrieve the best model
    # if best_model_name == 'DecisionTreeClassifier':
    #     best_model = dtc_model
    #     best_model.set_params(max_depth=dtc_max_depth)
    # elif best_model_name == 'RandomForestClassifier':
    #     best_model = rfc_model
    # else:
    #     best_model = lr_model
    models = [dtc_model, rfc_model, lr_model]
    params_list = [{'max_depth': dtc_max_depth}, rfc_params, {}]
    for best_model, best_model_params in zip(models,params_list):
        best_model.set_params(**best_model_params)
        
        # Refit the best model on the full training data
        best_model.fit(train_features, train_target)
        
        # Evaluate the best model on the test set
        best_model_test_score = best_model.score(test_features, test_target)
        #{models} replaced {best_model_name}
        print(f"Best Model: {models}\n \
              Optimal Hyperparameters: {best_model.get_params()}\n\
              Test Score: {best_model_test_score}")
    
        return best_model, best_model_test_score
    
# df = pd.read_csv("H:\My Drive\me\Jobs research\Tripleten Data Scientist\datasets\sprint 8 churn.csv") 
# RowNumber — data string index
# CustomerId — unique customer identifier
# Surname — surname
# CreditScore — credit score
# Geography — country of residence
# Gender — gender
# Age — age
# Tenure — period of maturation for a customer’s fixed deposit (years)
# Balance — account balance
# NumOfProducts — number of banking products used by the customer
# HasCrCard — customer has a credit card
# IsActiveMember — customer’s activeness
# EstimatedSalary — estimated salary
# Target
# Exited — сustomer has left

# target = df['Exited']  # Replace 'target_column' with the actual target column name
# features = df.drop(target.name, axis=1)  # Drop the target column from features
# best_model, best_score = model_picker(features, target, ['Gender'])