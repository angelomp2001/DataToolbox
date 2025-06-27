#everything starts here
import pandas as pd
import matplotlib.pyplot as plt
from data_explorers import view, see
from data_transformers import downsample, upsample, ordinal_encoder, missing_values, feature_scaler, categorical_encoder, data_splitter, data_transformer
from best_model_picker import best_model_picker, hyperparameter_optimizer
from sklearn.linear_model import LogisticRegression


# Extract and view
df = pd.read_csv('data/sprint 8 churn.csv')
#view(df)

#define target & identify ordinal categorical vars
target = df['Exited']
features = df.drop(target.name, axis = 1)

random_state = 99999


model_options = {
    'Regressions': {
        'LogisticRegression': LogisticRegression(random_state=random_state, solver='liblinear', max_iter=200)
    }
}


print(f'best_model_picker()...')
model_options, model_scores, optimized_hyperparameters, data = best_model_picker(
    features = features,
    target = target,
    n_rows = None,
    n_target_majority = None,
    ordinal_cols = None,
    random_state = random_state,
    model_options = model_options,
    split_ratio = (0.6, 0.2, 0.2),
    missing_values_method= None,
    fill_value = None,
    graph_scores=True
)
validation_scores = model_scores