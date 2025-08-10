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
categorical_cols = ['Geography', 'Gender']
data.encode_features(model_type='Machine Learning', categorical_cols=categorical_cols)
data.feature_scaler(show=True, column_names=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'])
data.split(split_ratio=(0.6, 0.2, 0.2), target_name=target.name, random_state=random_state).vectorize()
valid_features, valid_target = data.get_split(which='valid', columns='all')
train_features_vectorized, train_target_vectorized = data.get_vectorized(which='train', columns='all',show=True)
valid_features_vectorized, valid_target_vectorized = data.get_vectorized(which='valid', columns='all')

# raw
print(f'raw...')
best_scores_summary_df, _, best_scores_by_model, model_scores, transformed_data, model_options = best_model_picker(
    features = features,
    target = target,
    n_target_majority = None,
    n_target_minority = None,
    n_rows = None,
    ordinal_cols = None,
    random_state = random_state,
    model_options = {'Regressions': {'LogisticRegression': LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200)}}, #model_options,
    split_ratio = (0.6, 0.2, 0.2),
    missing_values_method= 'drop',
    fill_value = None,
    target_threshold = 0.5,
    metric=None,
    target_type='classification',
)


# ================================================
# ✅ Test 2: Fit using sklearn model and evaluate
# ================================================
print("\n=== Test 2: Sklearn Logistic Regression ===")
dm = DataModeler2()
# LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200)
dm.fit(
    train_features_vectorized=train_features_vectorized,
    train_target_vectorized=train_target_vectorized,
    model_type='classification',
    model_name='LogisticRegression'
    )
scores = dm.score(
    valid_features_vectorized=valid_features_vectorized,
    valid_target_vectorized=valid_target_vectorized,
    param_to_optimize= 'C',
    param_optimization_range= (1, 3),
    metric='Accuracy')


# # =====================================================
# # ✅ Test 3: Manual hyperparameter override before score
# # =====================================================
# print("\n=== Test 3: Manual Params Override ===")
# dm.score(
#     X_validation=validation_x,
#     y_validation=validation_y,
#     manual_params={"step_size": 0.005, "reg_weight": 0.0005},
#     verbose=True
# )

# # ==============================================================
# # ✅ Test 4: Optimize step_size using bisection search on GLM
# # ==============================================================

# print("\n=== Test 4: Optimize step_size for GLM ===")
# dm = DataModeler2(
#     random_state=42,
#     test_features=validation_x,
#     test_target=validation_y
# )
# dm.score(
#     X_validation=validation_x,
#     y_validation=validation_y,
#     optimize=True,
#     param_name='max_depth',
#     param_range=(5, 20),
#     metric='R2',
#     verbose=True
# )

# # =============================================================
# # ✅ Test 5: Optimize reg_weight using bisection search on GLM
# # =============================================================
# print("\n=== Test 5: Optimize reg_weight ===")
# dm = DataModeler2(
#     random_state=42,
#     test_features=validation_x,
#     test_target=validation_y
# )
# dm.score(
#     X_validation=validation_x,
#     y_validation=validation_y,
#     optimize=True,
#     param_name='reg_weight',
#     param_range=(0.00001, 0.1),
#     metric='f1',
#     verbose=True
# )

# #test 1
# model = DataModeler()
# model.fit(train_x, train_y, model_type='GLM')
# model.score(X_validation=validation_x, y_validation=validation_y)


# #test 2
# model.score(
#     X_validation=validation_x,
#     y_validation=validation_y,
#     manual_params={'step_size': 0.005, 'reg_weight': 0.001}
# )
# #test 3
# model.score(
#     X_validation=validation_x,
#     y_validation=validation_y,
#     optimize=True,
#     param_name='step_size',
#     param_range=(0.001, 0.1)
# )
# #test 4
# model.score(
#     X_validation=validation_x,
#     y_validation=validation_y,
#     optimize=True,
#     param_name='reg_weight',
#     param_range=(0.0001, 0.01),
#     #metric='mse'
# )

