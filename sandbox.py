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

# Extract and view
df = pd.read_csv('ds-automations/data/sprint 8 churn.csv')
# view(df) 

# Grab two random samples for testing
s1 = df[df['Gender'] == 'Male']['CreditScore']
s2 = df[df['Gender'] == 'Female']['CreditScore']

# Run the plot test with bootstrap
result = split_test_plot(s1, s2, bootstrap=True)

# Show output summary in console
print(result)


samples = bootstrap(df['CreditScore'], n=2, rows=df.shape[0], random_state=123)

split_test_plot(samples.loc[0], samples.loc[1])





