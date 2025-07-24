import pandas as pd
import numpy as np
from data_transformers import ordinal_encoder

def k_nearest(
    dataframe_or_features: pd.DataFrame,
    weights_dict: dict = None,
    new_row_dict: dict = None,
    k_nearest: int = 1
):
    """
    Find the k-nearest rows in the DataFrame based on distances.

    Parameters:
    - dataframe_or_features (pd.DataFrame): Input DataFrame of features.
    - weights_dict (dict): Optional weights for features.
    - new_row_dict (dict): Optional new row values for comparison.
    - k_nearest (int): Number of nearest rows to return.

    Returns:
    - pd.DataFrame: A DataFrame containing the k-nearest rows based on the specified calculations.

    Behavior:
    1. If `new_row` is None:
    - Calculates the 'average' row(s) in the DataFrame.
    
    2. If `new_row` is provided and len(new_row) == len(df):
    - Finds the row(s) in the DataFrame closest to `new_row`.
    
    3. If `new_row` is provided and len(new_row) < len(df):
    - Finds the closest missing value(s) from the DataFrame.
    """
    # check weights_dict match df
    if weights_dict:
        if not all(key in dataframe_or_features.columns for key in weights_dict.keys()):
            raise ValueError("Some keys in weights_dict do not match DataFrame columns.")
    
    # check new_row_dict match df
    if new_row_dict:
        feature_cols = [col for col in new_row_dict.keys() if col in dataframe_or_features.columns]
        
        # vectorize new_row_dict
        new_row_vector = np.array([new_row_dict[col] for col in feature_cols]).reshape(1, -1) # shape: (1, n)

    else:
        feature_cols = list(dataframe_or_features.columns)
    
    #feature_cols = list(new_row_dict.keys()) if new_row_dict else list(dataframe_or_features.columns)

    # vectorize df[feature_cols]
    dataframe_or_features, _ = ordinal_encoder(dataframe_or_features)
    df_matrix = dataframe_or_features[feature_cols].values # shape: (m, n)

    # initialize weights array
    if weights_dict is not None:
        # convert weight_dict to weight_array
        weights_array = np.array([weights_dict.get(col, 1) for col in feature_cols]) #shape: (features, )
    else:
        weights_array = np.ones(len(feature_cols)) #shape: (features, )

    'If new_row is None: find the most "average" row(s):'
    if new_row_dict is None:
        # initialize vars
        m_rows = len(df_matrix)
        total_distances = np.zeros(m_rows) #shape: (m_rows, )

        # Calculate distances between all rows
        for row in range(m_rows):
            # calculate difference in values between all rows
            diff_matrix = df_matrix - df_matrix[row]  # shape: (m_rows, features)
            
            # convert differences into weighted distances
            weighted_dist = np.sqrt(np.sum(weights_array * (diff_matrix ** 2), axis=1)) # shape: (m_rows, )

            # sum up distances by row
            total_distances[row] = weighted_dist.sum() # shape: (m_rows, )

        nearest_rows = np.argsort(total_distances)[:k_nearest] # shape: (k_nearest, )
            
        return dataframe_or_features.iloc[nearest_rows] #shape: (k_nearest, features)

    else:
        # Calculate distance to the new row
        diff_matrix = df_matrix - new_row_vector # shape: (m_rows, features)
        
        # convert differences into weighted distances
        distances = np.sqrt(np.sum(weights_array * (diff_matrix ** 2), axis=1)) # shape: (m_rows, )

        # save k-nearest distances to new_row
        k_nearest_distances = np.argsort(distances)[:k_nearest] # shape: (k_nearest, )

        'If new_row is provided and len(new_row) = len(df): find the row(s) in the DataFrame closest to new_row.'
        if set(new_row_dict.keys()) == set(dataframe_or_features.columns):          
            
            # lookup k-nearest rows using k-nearest distances
            k_nearest_rows = dataframe_or_features.iloc[k_nearest_distances] # shape: (k_nearest, features)
            return k_nearest_rows
        
        else:
            'If new_row is provided and len(new_row) < len(df): find the closest missing value(s).'
            # get cols to predict
            y_hats = [col for col in dataframe_or_features.columns if col not in new_row_dict]

            # lookup y_yats using k-nearest distances
            k_nearest_y_hats = dataframe_or_features.iloc[k_nearest_distances][y_hats]
            
            return k_nearest_y_hats


