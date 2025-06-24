#everything starts here
from view import view
from data_transformer import data_transformer
import pandas as pd
from best_model import best_model

# Extract and view
df = pd.read_csv('data/sprint 8 churn.csv')
#view(df)

#define target & identify ordinal categorical vars
target = df['Exited']

# Transform data
train_features, train_target, valid_features, valid_target, test_features, test_target = data_transformer(
        df, 
        target=target.name, 
        split_ratio=(0.6, 0.2, 0.2), 
        random_state=412345)

#best_model, best_model_test_score = best_model(features, target, ordinal_cols=[])