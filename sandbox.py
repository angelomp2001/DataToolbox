#everything starts here
import pandas as pd
import matplotlib.pyplot as plt
from data_explorers import view, see
from data_transformers import downsample, upsample, ordinal_encoder, missing_values, feature_scaler, categorical_encoder, data_splitter, data_transformer
from best_model_picker import best_model_picker, hyperparameter_optimizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from metrics import categorical_scorer
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split



# Extract and view
df = pd.read_csv('data/sprint 8 churn.csv')
#view(df)

#define target & identify ordinal categorical vars
target = df['Exited']
features = df.drop(target.name, axis = 1)

random_state = 99999


model_options = {
    'Machine Learning': {
        'RandomForestClassifier': RandomForestClassifier(random_state=random_state)
    }
}


# print(f'best_model_picker()...')
# model_options, model_scores, optimized_hyperparameters, data = best_model_picker(
#     features = features,
#     target = target,
#     n_rows = None,
#     n_target_majority = None,
#     ordinal_cols = None,
#     random_state = random_state,
#     model_options = model_options,
#     split_ratio = (0.6, 0.2, 0.2),
#     missing_values_method= None,
#     fill_value = None,
#     graph_scores=True,
#     metric='F1'
# )
# validation_scores = model_scores


# Testing using Logistic Regression
if __name__ == "__main__":
    # Create synthetic binary classification data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict probabilities for the test set (probability for class 1)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Convert target and predictions to pandas Series
    y_test_series = pd.Series(y_test)
    y_pred_series = pd.Series(y_pred_prob)
    
    # Compute evaluation metrics using the categorical_scorer
    accuracy, precision, recall, f1 = categorical_scorer(y_test_series, y_pred_series)
    
    print("Evaluation metrics for Logistic Regression:")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)