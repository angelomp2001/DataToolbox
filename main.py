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
model_scores, optimized_hyperparameters, data, model_options = best_model_picker(
    features = features,
    target = target,
    n_rows = None,
    n_target_majority = None,
    ordinal_cols = None,
    random_state = random_state,
    model_options = None,
    split_ratio = (0.6, 0.2, 0.2),
    missing_values_method= None,
    fill_value = None
)

print(f'testing all models on test data...')
model_scores, _, _, model_options = best_model_picker(
    features = data[2],
    target = data[3],
    test_features= data[4],
    test_target= data[5],
    random_state = random_state,
    model_options= model_options,
    model_params = optimized_hyperparameters,
)