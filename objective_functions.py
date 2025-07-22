import pandas as pd
import numpy as np



def k_nearest(
        dataframe_or_features: pd.DataFrame,
        weights: dict = None,
        new_row: dict = None,
        k_nearest: int = 1
):
    '''
    nearest row(s):
        input: df, (series)
        output: row(s) in df that's nearest to all other rows in df - the most 'average' row(s).
    nearest neighbor(s):
        input: df, series, new_row
        output: nearest target value (missing in new_row dict) that matches new_row
    '''
    # nearest row(s):
    if new_row is None:
        # Vectorize df
        D = dataframe_or_features.values
        weights = np.array(list(new_row.values())).reshape(1, -1) if weights is not None else np.ones(D.shape[1]) # weights = weights.values if weights is not None else np.ones(D.shape[1]) # D.shape[1] or D.shape[0]

        # Distance matrix
        distances = np.zeros((len(D), len(D)))
        for m in range(len(D)):
            for n in range(len(D)):
                difference = D[m] - D[n]
                distances[m, n] = np.dot(difference, difference) # ** 0.5 won't matter if returning rank

        # Apply weights
        distances = weights * distances
        # sum up to get a vector of distances
        total_distances = distances.sum(axis=1)

        # concat distances to D
        D = np.concatenate((D, total_distances[:, np.newaxis]), axis=1)

        # return k nearest rows
        nearest_rows = np.argsort(D[:, -1])[:k_nearest]
        
        return dataframe_or_features.iloc[nearest_rows]

    else:
        # nearest neighbor(s):
        # Vectorize df
        D = dataframe_or_features[list(new_row.keys())].values
        weights = weights.values if weights is not None else np.ones(D.shape[1])
        new_row_vector = np.array(list(new_row.values())).reshape(1, -1)

        # Calculate distances
        distances = np.sqrt(np.sum(weights * ((D - new_row_vector) ** 2), axis=1))
        
        # return k nearest based on nearest distances
        nearest_rows = np.argsort(distances)[:k_nearest]
        nearest_k = dataframe_or_features.iloc[nearest_rows]
        return nearest_k



        
def chat_k_nearest(
    dataframe_or_features: pd.DataFrame,
    weights_dict: dict = None,
    new_row_dict: dict = None,
    k_nearest: int = 1
):
    """
    If new_row is None: find the most 'average' row(s):
        input: df, (series)
        output: row(s) in df that's nearest to all other rows in df - the most 'average' row(s).
    If new_row is provided: find the row(s) in the DataFrame closest to new_row.
        input: df, series, new_row
        output: nearest row(s) in df that matches new_row
    If new_row is provided, but missing value(s): find the closest missing value(s).
        input: df, series, new_row
        output: nearest target value (missing in new_row dict) that matches new_row 
    

    """
    # check weights match df
    if weights_dict is not None:
    key in dataframe_or_features for key in weights_dict.keys()
            
    # define features
    feature_cols = list(new_row_dict.keys()) if new_row_dict else list(dataframe_or_features.columns)

    # vectorize df
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
        total_distances = np.zeros(m_rows) #shape: (m, )

        # Calculate distances between all rows
        for row in range(m_rows):
            diff_matrix = df_matrix - df_matrix[row]  # shape: (m_rows, features)
            weighted_dist = np.sqrt(np.sum(weights_array * (diff_matrix ** 2), axis=1)) # shape: (m_rows, )

            total_distances[row] = weighted_dist.sum() # shape: (m_rows, )

        nearest_rows = np.argsort(total_distances)[:k_nearest] # shape: (k_nearest, )
        return dataframe_or_features.iloc[nearest_rows] #shape: (k_nearest, features)

    else:
        'If new_row is provided: find the row(s) in the DataFrame closest to new_row.'
        if len(new_row_dict) == len(feature_cols):
        
            # Calculate distance to the new row
            new_row_vector = np.array([new_row_dict[col] for col in feature_cols]).reshape(1, -1) # shape: (1, n)
            diff_matrix = df_matrix - new_row_vector # shape: (m_rows, features)
            distances = np.sqrt(np.sum(weights_array * (diff_matrix ** 2), axis=1)) # shape: (m_rows, )
            nearest_rows = np.argsort(distances)[:k_nearest] # shape: (k_nearest, )
            return dataframe_or_features.iloc[nearest_rows] # shape: (k_nearest, features)
        else:
            'If new_row is provided, but missing value(s): find the closest missing value(s).'

            # Calculate distance to the new row
