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
train_features_vectorized, train_target_vectorized = data.get_vectorized(which='train', columns='all')
features_vectorized, target_vectorized = data.get_vectorized(which='df', columns='both', show=True)
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


# test 1 & 11: .fit() train features and target, model_type, model_name
data = DataModeler2(
    valid_features_vectorized=valid_features_vectorized,
    valid_target_vectorized=valid_target_vectorized
)
print('test 1')
data.fit(
    train_features_vectorized=train_features_vectorized,
    train_target_vectorized=train_target_vectorized,
    model_type='classification',
    model_name='RandomForestClassifier', #DecisionTreeClassifier(random_state=random_state)
    model_params={'random_state': random_state},
    )

# test 2: .fit() train features and target, no model_type, no model_name
# this will use the default model options
# print('test 2')
# data.fit(
#     train_features_vectorized=train_features_vectorized,
#     train_target_vectorized=train_target_vectorized,
#     model_type='classification',
#     model_name='LogicticRegression'
#     )

# test 3: no specificed models, and does default
# print('test 3')
# data.fit(
#     train_features_vectorized=train_features_vectorized,
#     train_target_vectorized=train_target_vectorized
# )

# test list input
# print('test 3.5')
# data.fit(
#     train_features_vectorized=train_features_vectorized,
#     train_target_vectorized=train_target_vectorized,
#     model_name=['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier']
# )

# test 4: .score with valid features and target
print('test 4')
data.score(
    valid_features_vectorized=valid_features_vectorized,
    valid_target_vectorized=valid_target_vectorized,
    metric='F1',
    manual_params={'max_depth': 10},
    verbose=True  
)

#test 5: .score with valid features and target, manual params
# print('test 5')
# data.score(
#     valid_features_vectorized=valid_features_vectorized,
#     valid_target_vectorized=valid_target_vectorized,
#     metric='F1'
# )

# # test 6: .score with valid features and target, no manual params or metric
# print('test 6')
# data.score(
#     valid_features_vectorized=valid_features_vectorized,
#     valid_target_vectorized=valid_target_vectorized
# )

# # test 7: .score return all metrics 
# print('test 7')
# data.score(
#     valid_features_vectorized=valid_features_vectorized,
#     valid_target_vectorized=valid_target_vectorized,
# )

# # test 8: no data, use self
# print('test 8')
# data.score()

# test 9: chain with manual params
# print('test 9')
# data.score(
#     param_to_optimize= 'n_estimators',
#     param_optimization_range=(1, 100),
#     metric='F1'
# )

# # test 10: chain with manual params and metric
# print('test 10')
# data.score(
#     manual_params={'max_depth': 5},
#     param_to_optimize='n_estimators',
#     param_optimization_range=(10, 100),
#     metric='ROC AUC'
# )
