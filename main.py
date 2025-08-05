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
from DataModeler import DataModeler

# Extract and view
df = pd.read_csv('data/sprint 8 churn.csv')

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



#raw
# print(f'raw...')
# best_scores_summary_df, _, best_scores_by_model, model_scores, transformed_data, model_options = best_model_picker(
#     features = features,
#     target = target,
#     n_target_majority = None,
#     n_target_minority = None,
#     n_rows = None,
#     ordinal_cols = None,
#     random_state = random_state,
#     model_options = model_options,
#     split_ratio = (0.6, 0.2, 0.2),
#     missing_values_method= 'drop',
#     fill_value = None,
#     target_threshold = 0.5,
#     metric=None,
#     target_type='classification'
# )

data = DataProcessor(df)
data.missing_values(missing_values_method='drop', fill_value=None)
categorical_cols = ['Surname', 'Geography', 'Gender']
data.encode_features(model_type='Machine Learning', categorical_cols=categorical_cols)
data.feature_scaler()
train_features, train_target, validation_features, validation_target, test_features, test_target = data.split(split_ratio=(0.6, 0.2, 0.2), target=target.name)
train_x, train_y, = data.vectorize(features=train_features, target=train_target)


model = DataModeler()
model.fit(models = model_options, features=train_x, target=train_y)
