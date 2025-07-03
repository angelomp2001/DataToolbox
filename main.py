#everything starts here
import pandas as pd
import matplotlib.pyplot as plt
from data_explorers_v2 import view, see
from data_transformers_v2 import downsample, upsample, ordinal_encoder, missing_values, feature_scaler, categorical_encoder, data_splitter, data_transformer
from best_model_picker_v2 import optimizer, best_model_picker
from metrics_v2 import categorical_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sys
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
best_scores_summary_df, optimized_hyperparameters, best_scores_by_model, model_scores, transformed_data, model_options = best_model_picker(
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
    target_threshold = 0.5,
    metric=metric,
    target_type='classification'
)
raw_validation_scores = model_scores

#sys.exit()
#pdb.set_trace() # pauses code

'''
raw:
  Metric Name  Threshold  Best Score              Model Name
0    Accuracy        0.5    0.857614  RandomForestClassifier
1   Precision        0.5    0.798995  RandomForestClassifier
2      Recall        0.5    0.505291  DecisionTreeClassifier
3          F1        0.5    0.551127  RandomForestClassifier

                Model Name  Threshold     Metric     Score
0       LogisticRegression        0.5   Accuracy  0.792194
1       LogisticRegression        0.5  Precision  0.000000
2       LogisticRegression        0.5     Recall  0.000000
3       LogisticRegression        0.5         F1  0.000000
4   DecisionTreeClassifier        0.5   Accuracy  0.800440
5   DecisionTreeClassifier        0.5  Precision  0.520436
6   DecisionTreeClassifier        0.5     Recall  0.505291
7   DecisionTreeClassifier        0.5         F1  0.512752
8   RandomForestClassifier        0.5   Accuracy  0.857614
9   RandomForestClassifier        0.5  Precision  0.798995
10  RandomForestClassifier        0.5     Recall  0.420635
11  RandomForestClassifier        0.5         F1  0.551127
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
best_scores_summary_df, optimized_hyperparameters, best_scores_by_model, model_scores, transformed_data, model_options = best_model_picker(
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
    target_threshold = 0.5,
    metric=metric,
    target_type='classification'
)

lr_balanced_validation_scores = model_scores

'''
  Metric Name  Threshold  Best Score          Model Name
0    Accuracy        0.5    0.713029  LogisticRegression
1   Precision        0.5    0.386435  LogisticRegression
2      Recall        0.5    0.648148  LogisticRegression
3          F1        0.5    0.484190  LogisticRegression


'''

#pdb.set_trace() # pauses code

# upsampling
print(f'upsampling...')
best_scores_summary_df, optimized_hyperparameters, best_scores_by_model, model_scores, transformed_data, model_options = best_model_picker(
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
    target_threshold = 0.5,
    metric=metric,
    target_type='classification'
)
upsampling_scores = model_scores
#pdb.set_trace() # pauses code

'''
  Metric Name  Threshold  Best Score              Model Name
0    Accuracy        0.5    0.805296  RandomForestClassifier
1   Precision        0.5    0.732558  RandomForestClassifier
2      Recall        0.5    0.548649  DecisionTreeClassifier
3          F1        0.5    0.601911  RandomForestClassifier

                Model Name  Threshold     Metric     Score
0       LogisticRegression        0.5   Accuracy  0.711838
1       LogisticRegression        0.5  Precision  0.000000
2       LogisticRegression        0.5     Recall  0.000000
3       LogisticRegression        0.5         F1  0.000000
4   DecisionTreeClassifier        0.5   Accuracy  0.725078
5   DecisionTreeClassifier        0.5  Precision  0.521851
6   DecisionTreeClassifier        0.5     Recall  0.548649
7   DecisionTreeClassifier        0.5         F1  0.534914
8   RandomForestClassifier        0.5   Accuracy  0.805296
9   RandomForestClassifier        0.5  Precision  0.732558
10  RandomForestClassifier        0.5     Recall  0.510811
11  RandomForestClassifier        0.5         F1  0.601911
'''
# downsampling
print(f'downsampling...')
best_scores_summary_df, optimized_hyperparameters, best_scores_by_model, model_scores, transformed_data, model_options = best_model_picker(
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
    target_threshold = 0.5,
    target_type='classification',
    metric=metric
)
downsampling_scores = model_scores
#pdb.set_trace() # pauses code
'''
  Metric Name  Threshold  Best Score              Model Name
0    Accuracy        0.5    0.857339  RandomForestClassifier
1   Precision        0.5    0.877095  RandomForestClassifier
2      Recall        0.5    0.839572  RandomForestClassifier
3          F1        0.5    0.857923  RandomForestClassifier

                Model Name  Threshold     Metric     Score
0       LogisticRegression        0.5   Accuracy  0.559671
1       LogisticRegression        0.5  Precision  0.564792
2       LogisticRegression        0.5     Recall  0.617647
3       LogisticRegression        0.5         F1  0.590038
4   DecisionTreeClassifier        0.5   Accuracy  0.805213
5   DecisionTreeClassifier        0.5  Precision  0.813514
6   DecisionTreeClassifier        0.5     Recall  0.804813
7   DecisionTreeClassifier        0.5         F1  0.809140
8   RandomForestClassifier        0.5   Accuracy  0.857339
9   RandomForestClassifier        0.5  Precision  0.877095
10  RandomForestClassifier        0.5     Recall  0.839572
11  RandomForestClassifier        0.5         F1  0.857923
'''
# threshold adjustment
print(f'threshold adjustment...')
best_scores_summary_df, optimized_hyperparameters, best_scores_by_model, model_scores, transformed_data, model_options = best_model_picker(
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
    metric=metric,
    target_type='classification',
    target_threshold=None,
    #model_params={'Machine Learning': {'DecisionTreeClassifier': {'max_depth': 16}}}
)

#threshold_scores = best_scores_summary_df

'''
  Metric Name  Threshold  Best Score              Model Name
0    Accuracy       0.47    0.856515  RandomForestClassifier
1   Precision       0.83    1.000000  RandomForestClassifier
2      Recall       0.01    1.000000      LogisticRegression
3          F1       0.39    0.600000  RandomForestClassifier

                Model Name  Threshold     Metric     Score
0       LogisticRegression       0.31   Accuracy  0.793293
1       LogisticRegression       0.31  Precision  0.750000
2       LogisticRegression       0.01     Recall  1.000000
3       LogisticRegression       0.19         F1  0.359313
4   DecisionTreeClassifier       0.01   Accuracy  0.800440
5   DecisionTreeClassifier       0.01  Precision  0.520436
6   DecisionTreeClassifier       0.01     Recall  0.505291
7   DecisionTreeClassifier       0.01         F1  0.512752
8   RandomForestClassifier       0.47   Accuracy  0.856515
9   RandomForestClassifier       0.83  Precision  1.000000
10  RandomForestClassifier       0.01     Recall  0.997354
11  RandomForestClassifier       0.39         F1  0.600000
'''

"4. Perform the final testing."

'''
raw:
  Metric Name  Best Score              Model Name
0    Accuracy    0.857614  RandomForestClassifier
1   Precision    0.798995  RandomForestClassifier
2      Recall    0.505291  DecisionTreeClassifier
3          F1    0.551127  RandomForestClassifier

lr balance:
  Metric Name  Best Score          Model Name
0    Accuracy    0.713029  LogisticRegression
1   Precision    0.386435  LogisticRegression
2      Recall    0.648148  LogisticRegression
3          F1    0.484190  LogisticRegression

upsample:
  Metric Name  Best Score              Model Name
0    Accuracy    0.805296  RandomForestClassifier
1   Precision    0.732558  RandomForestClassifier
2      Recall    0.548649  DecisionTreeClassifier
3          F1    0.601911  RandomForestClassifier

downsample:
  Metric Name  Best Score              Model Name
0    Accuracy    0.857339  RandomForestClassifier
1   Precision    0.877095  RandomForestClassifier
2      Recall    0.839572  RandomForestClassifier
3          F1    0.857923  RandomForestClassifier

threshold:
  Metric Name  Threshold  Best Score              Model Name
0    Accuracy       0.47    0.856515  RandomForestClassifier
1   Precision       0.83    1.000000  RandomForestClassifier
2      Recall       0.01    1.000000      LogisticRegression
3          F1       0.39    0.600000  RandomForestClassifier

                Model Name  Threshold     Metric     Score
0       LogisticRegression       0.31   Accuracy  0.793293
1       LogisticRegression       0.31  Precision  0.750000
2       LogisticRegression       0.01     Recall  1.000000
3       LogisticRegression       0.19         F1  0.359313
4   DecisionTreeClassifier       0.01   Accuracy  0.800440
5   DecisionTreeClassifier       0.01  Precision  0.520436
6   DecisionTreeClassifier       0.01     Recall  0.505291
7   DecisionTreeClassifier       0.01         F1  0.512752
8   RandomForestClassifier       0.47   Accuracy  0.856515
9   RandomForestClassifier       0.83  Precision  1.000000
10  RandomForestClassifier       0.01     Recall  0.997354
11  RandomForestClassifier       0.39         F1  0.600000

Logistic Regression never performed the best. 
Upsampling reduced accuracy and precision but improved F1. 
Downsampling improved all scores. 
Optimizing the threshold improved precision and recall, but lowered F1.  Maybe ML hyperparameters were optimized for a 0.5 threshold? 
'''

"with target threshold at 0.5"
best_scores_summary_df, optimized_hyperparameters, best_scores_by_model, model_scores, transformed_data, model_options = best_model_picker(
    features = features,
    target = target,
    n_rows = 4037,
    n_target_majority = .2*10000,
    n_target_minority = 2000,
    ordinal_cols = None,
    random_state = random_state,
    model_options = None,
    split_ratio = (0.6, 0.2, 0.2),
    missing_values_method= 'drop',
    fill_value = None,
    metric=metric,
    target_type='classification',
    target_threshold=0.5,
)

'''
  Metric Name  Threshold  Best Score              Model Name
0    Accuracy        0.5    0.894595  RandomForestClassifier
1   Precision        0.5    0.927746  RandomForestClassifier
2      Recall        0.5    0.858289  RandomForestClassifier
3          F1        0.5    0.891667  RandomForestClassifier
'''
"optimized target threshold"
best_scores_summary_df, optimized_hyperparameters, best_scores_by_model, model_scores, transformed_data, model_options = best_model_picker(
    features = features,
    target = target,
    n_rows = 4037,
    n_target_majority = .2*10000,
    n_target_minority = 2000,
    ordinal_cols = None,
    random_state = random_state,
    model_options = None,
    split_ratio = (0.6, 0.2, 0.2),
    missing_values_method= 'drop',
    fill_value = None,
    metric=metric,
    target_type='classification',
    target_threshold=None,
)
'''
  Metric Name  Threshold  Best Score              Model Name
0    Accuracy       0.43    0.904054  RandomForestClassifier
1   Precision       0.71    1.000000      LogisticRegression
2      Recall       0.01    1.000000      LogisticRegression
3          F1       0.39    0.905759  RandomForestClassifier

Best model was RandomForestClassifier:
- downsampled to 4027 rows
- minority upsampled to 2000 rows (50%)
- sequentially optimized hyperparameters
- optimized threshold to 0.39.  
'''