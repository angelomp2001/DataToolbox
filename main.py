#everything starts here
import pandas as pd
import matplotlib.pyplot as plt
from data_explorers import view, see
from data_transformers import downsample, upsample, ordinal_encoder, missing_values, feature_scaler, categorical_encoder, data_splitter, data_transformer
from best_model_picker import best_model_picker, hyperparameter_optimizer


# Extract and view
df = pd.read_csv('data/sprint 8 churn.csv')
#view(df)

#define target & identify ordinal categorical vars
target = df['Exited']
features = df.drop(target.name, axis = 1)

random_state = 99999


print(f'best_model_picker()...')
model_options, model_scores, optimized_hyperparameters, data = best_model_picker(
    features = features,
    target = target,
    ordinal_cols = None,
    random_state = random_state,
    model_options = None,
    split_ratio = (0.6, 0.2, 0.2),
    missing_values_method= None,
)
validation_scores = model_scores

print(f'testing all models on test data...')
model_options, model_scores, _, _ = best_model_picker(
    features = data[2],
    target = data[3],
    test_features= data[4],
    test_target= data[5],
    random_state = random_state,
    model_options= model_options,
    model_params = optimized_hyperparameters,
)

print(f'validation scores: {validation_scores}')
print(f'test scores: {model_scores}')
# output is 100% accurate, so it's being fitted and tested on the same data.  I need to fit on training data, and test on test data.  
# how do I fit on training data in the second interation? I'm only submitting test data.  so it's being  