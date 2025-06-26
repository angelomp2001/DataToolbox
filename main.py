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

print(f'best_model_picker()...')
best_model_name, best_model_score, optimized_hyperparameters, transformed_data = best_model_picker(
    features = features,
    target = target,
    ordinal_cols = None,
    random_state = 99999,
    model_options = None,
    split_ratio = (0.6, 0.2, 0.2),
    missing_values_method= None,
)

print(f'testing all models on test data...')
best_model_name, best_model_score, optimized_hyperparameters, _ = best_model_picker(
    features = transformed_data[4],
    target = transformed_data[5],
    ordinal_cols = None,
    random_state = 99999,
)