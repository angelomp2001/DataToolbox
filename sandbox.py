#everything starts here
import pandas as pd
import matplotlib.pyplot as plt
from data_explorers import view, see
from data_transformers import downsample, upsample, ordinal_encoder, missing_values, feature_scaler, categorical_encoder, data_splitter, data_transformer
from best_model_picker import optimizer, best_model_picker
from model_scorer import categorical_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sys
import pdb
import inspect
from DataProcessor import DataProcessor
from DataModeler2 import DataModeler2

# Extract and view
df = pd.read_csv('data/sprint 8 churn.csv')
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)  # Drop unnecessary columns

# sample df
df = df[['CreditScore','Age', 'Tenure','Balance','NumOfProducts','HasCrCard', 'IsActiveMember','EstimatedSalary', 'Exited']]

'''
# columns=['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
'''

#define target & identify ordinal categorical vars
target = df['Exited']
features = df.drop(target.name, axis = 1)
random_state = 99999

model_options = {
    'Regressions': {
        'LogisticRegression': LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200)
    },
    'Machine Learning': {
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=random_state),
        'RandomForestClassifier': RandomForestClassifier(random_state=random_state),
        
    }
}

data = DataProcessor(df)
data.missing_values(missing_values_method='drop', fill_value=None)
# categorical_cols = ['Geography', 'Gender']
# data.encode_features(model_type='Machine Learning', categorical_cols=categorical_cols)
#data.feature_scaler(show=True, column_names=['EstimatedSalary'])
data.split(split_ratio=(0.6, 0.2, 0.2), target_name=target.name, random_state=random_state).vectorize()
# for old functions
features, target = data.get_split(which='train', columns='both')
train_features_vectorized, train_target_vectorized = data.get_vectorized(which='train', columns='both')
features_vectorized, target_vectorized = data.get_vectorized(which='df', columns='both')
valid_features_vectorized, valid_target_vectorized = data.get_vectorized(which='valid', columns='both')
# # raw
# print(f'raw...')
# best_scores_summary_df, _, best_scores_by_model, model_scores, transformed_data, model_options = best_model_picker(
#     features = features,
#     target = target,
#     n_target_majority = None,
#     n_target_minority = None,
#     n_rows = None,
#     ordinal_cols = None,
#     random_state = random_state,
#     model_options = {'Regressions': {'LogisticRegression': LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200)}}, #model_options,
#     split_ratio = (0.6, 0.2, 0.2),
#     missing_values_method= 'drop',
#     fill_value = None,
#     target_threshold = 0.5,
#     metric=None,
#     target_type='classification',
# )

Logistic_Regression = DataModeler2()
print('test 1')
Logistic_Regression.fit(
    train_features_vectorized=train_features_vectorized,
    train_target_vectorized=train_target_vectorized,
    model_type='classification',
    model_name=['LogisticRegression'], #'DecisionTreeClassifier', 'RandomForestClassifier', 'xgboost', 'lgbm'],
    model_params={'solver': 'liblinear', 'max_iter': 200, 'class_weight': 'balanced'} # ,'learning_rate': 0.1, 'eval_metric': 'logloss', 'tree_method': 'auto'}
    )
Logistic_Regression.score(
        valid_features_vectorized=valid_features_vectorized,
        valid_target_vectorized=valid_target_vectorized,
        metric='ROC AUC',
        verbose=False
    )
tree_models = DataModeler2()
tree_models.fit(
    train_features_vectorized=train_features_vectorized,
    train_target_vectorized=train_target_vectorized,
    model_type='classification',
    model_name=['DecisionTreeClassifier', 'RandomForestClassifier', 'xgboost']
    )
tree_models.score(
    valid_features_vectorized=valid_features_vectorized,
    valid_target_vectorized=valid_target_vectorized,
    param_to_optimize='max_depth',
    param_optimization_range=(5, 20),
    metric='ROC AUC'
)
tree_models.score(
    param_to_optimize='n_estimators',
    param_optimization_range=(50, 200),
    metric='ROC AUC'
    )
