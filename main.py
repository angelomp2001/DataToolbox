#everything starts here
import pandas as pd
import matplotlib.pyplot as plt
from data_explorers_v2 import view, see
from data_transformers_v2 import downsample, upsample, ordinal_encoder, missing_values, feature_scaler, categorical_encoder, data_splitter, data_transformer
from best_model_picker_v2 import best_model_picker
from metrics_v2 import categorical_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# import sys
import pdb
'''
Project instructions
1. Download and prepare the data. Explain the procedure.
2. Examine the balance of classes. Train the model without taking into account the imbalance. Briefly describe your findings.
3. Improve the quality of the model. Make sure you use at least two approaches to fixing class imbalance. Use the training set to pick the best parameters. Train different models on training and validation sets. Find the best one. Briefly describe your findings.
4. Perform the final testing.
'''

"1. Download and prepare the data. Explain the procedure."
# Extract and view
df = pd.read_csv('data/sprint 8 churn.csv')

"2. Examine the balance of classes"
# I will now view the data
# view(df)

'''
# I'll keep the header names, encode categorical, ['Exit'] has minority of 20%, which I think is fine. especially out of 10k rows.   
# columns=['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
'''
# I will visually view the data
# see(df)
# With 10000 rows, these graphs are not very useful to me. Nothing stands out like a missing data pattern, etc.  
# I will transform the data as follows:
# encode categorical
# NOTE: ['Exit'] has minority of 20% and will stay that way:

#define target & identify ordinal categorical vars
target = df['Exited']
features = df.drop(target.name, axis = 1)

random_state = 99999

model_options = {
            'Regressions': {
                'LogisticRegression': LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200)},
            'Machine Learning': {
                'DecisionTreeClassifier': DecisionTreeClassifier(random_state=random_state),
                'RandomForestClassifier': RandomForestClassifier(random_state=random_state),
                
            }
        }

metric = None

#raw
print(f'raw...')
model_options, model_scores, optimized_hyperparameters, data = best_model_picker(
    features = features,
    target = target,
    n_target_majority = None,
    n_target_minority = None,
    n_rows = None,
    ordinal_cols = None,
    random_state = random_state,
    model_options = model_options,
    split_ratio = (0.6, 0.2, 0.2),
    missing_values_method= 'drop',
    fill_value = None,
    threshold = 0.5,
    metric=metric
)
raw_validation_scores = model_scores

#sys.exit()
pdb.set_trace() # pauses code

'''
               Model Name  Accuracy
0      LogisticRegression  0.792194
1  DecisionTreeClassifier  0.800440
2  RandomForestClassifier  0.857614 <- best model
'''


"3. Improve the quality of the model."

'''
1. class weight adjustment
2. upsample
3. downsample
4. Threshold adjustment
'''

# class weight adjustment logistic regression
model_options = {
    'Regressions': {
        'LogisticRegression': LogisticRegression(random_state=12345, solver='liblinear', class_weight='balanced')
    }
}

print(f'class weight adjustment...')
model_options, model_scores, optimized_hyperparameters, data = best_model_picker(
    features = features,
    target = target,
    n_rows = None,
    n_target_majority = None,
    ordinal_cols = None,
    random_state = random_state,
    model_options = model_options,
    split_ratio = (0.6, 0.2, 0.2),
    missing_values_method= 'drop',
    fill_value = None,
    threshold = 0.5,
    metric=metric
)

lr_balanced_validation_scores = model_scores

'''
Model Name    LogisticRegression
Accuracy                0.713029
'''

pdb.set_trace() # pauses code

# upsampling
print(f'upsampling...')
model_options, model_scores, optimized_hyperparameters, data = best_model_picker(
    features = features,
    target = target,
    n_rows = None,
    n_target_majority = 5000,
    ordinal_cols = None,
    random_state = random_state,
    model_options = None,
    split_ratio = (0.6, 0.2, 0.2),
    missing_values_method= 'drop',
    fill_value = None,
    threshold = 0.5,
    metric=metric
)
upsampling_scores = model_scores
pdb.set_trace() # pauses code
# downsampling
print(f'downsampling...')
model_options, model_scores, optimized_hyperparameters, data = best_model_picker(
    features = features,
    target = target,
    n_rows = 4000,
    n_target_majority = .2*10000,
    ordinal_cols = None,
    random_state = random_state,
    model_options = None,
    split_ratio = (0.6, 0.2, 0.2),
    missing_values_method= 'drop',
    fill_value = None,
    threshold = 0.5,
    metric=metric
)
downsampling_scores = model_scores
pdb.set_trace() # pauses code
# threshold adjustment
print(f'threshold adjustment...')
model_options, model_scores, optimized_hyperparameters, data = best_model_picker(
    features = features,
    target = target,
    n_rows = None,
    n_target_majority = None,
    ordinal_cols = None,
    random_state = random_state,
    model_options = None,
    split_ratio = (0.6, 0.2, 0.2),
    missing_values_method= 'drop',
    fill_value = None,
    metric=metric
)

threshold_scores = model_scores

"4. Perform the final testing."


