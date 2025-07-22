#everything starts here
import pandas as pd
import matplotlib.pyplot as plt
from data_explorers import view, see
from data_transformers import downsample, upsample, ordinal_encoder, missing_values, feature_scaler, categorical_encoder, data_splitter, data_transformer, bootstrap
from best_model_picker import optimizer, best_model_picker
from model_scorer import categorical_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from model_scorer import categorical_scorer
import sys
import pdb
import inspect
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from H0_testing import split_test_plot
import scipy.stats as st
from statsmodels.stats.power import TTestIndPower
from objective_functions import k_nearest, chat_k_nearest
import tqdm

# maximize terminal output display
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'left')


# Extract and view
df = pd.read_csv('data/sprint 8 churn.csv')
# view(df)


sub_df = df.loc[:4, ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']]
print(sub_df.head())
# weights = None
# D = sub_df.values
# weights = weights.values if weights is not None else np.ones(len(D))
# new_row = None


# distances = np.zeros((len(D), len(D)))
# print(D,'\n')
# for m in range(len(D)):
#     for n in range(len(D)):
#         print(f"Comparing row {m} and row {n}:")
#         print("Row m:", D[m])
#         print("Row n:", D[n])
#         difference = D[m] - D[n]
#         print("Difference:", difference)
#         distances[m, n] = np.dot(difference, difference)

#     print(distances)
weights = {
    'CreditScore': 0,
    'Age': 99999,
    'Tenure': .0,
    'Balance': .0,
    'NumOfProducts': .0,
    'HasCrCard': .0,
    'IsActiveMember': .0,
    'EstimatedSalary': .0,
    'Exited': .0,
}

new_row = {
    'CreditScore': 619,
    'Age': 42,
    'Tenure': 2,
    'Balance': 0,
    'NumOfProducts': 1,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 101348.88,
    'Exited': 0,
}
output = chat_k_nearest(dataframe_or_features=sub_df, weights=weights, new_row=None, k_nearest=1)
print(output)
# # Grab two random samples for testing
# s1 = df[df['Gender'] == 'Male']['CreditScore']
# s2 = df[df['Gender'] == 'Female']['CreditScore']

# # Run the plot test with bootstrap
# result = split_test_plot_test(s1, s2, bootstrap=True)

# # Show output summary in console
# print(result)


# samples = bootstrap(df['CreditScore'], n=2, rows=df.shape[0], random_state=123)

# split_test_plot_test(samples.loc[0], samples.loc[1])





