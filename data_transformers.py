# from data_transformer import downsample, upsample, ordinal_encoder, missing_values, feature_scaler, categorical_encoder, data_splitter, data_transformer

# input df and requirements, outputs processed df
# processing: downsample, upsample, ordinal_encoder, missing_values, feature_scaler, categorical_encoder, data_splitter, data_transformer

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, shuffle
from typing import Optional
import numpy as np




def bootstrap(
        data: pd.DataFrame,
        n: int = 100,
        rows: int = None,
        frac: float = None,
        replace: bool = True,
        weights: Optional[np.ndarray] = None,
        random_state: int = None,
        axis: str = 'index'
    ):
    """
    Create n bootstrap samples from the provided data, always resetting the index.
    
    Parameters:
      data: DataFrame or Series.
      n: Number of bootstrap samples.
      rows: Number of rows (or elements) to sample per bootstrap sample.
            If provided, 'frac' is ignored.
      frac: Fraction of data to sample if rows is None.
      replace: Sample with replacement if True.
      weights: Weights for sampling; default None gives equal probability.
      random_state: Seed for randomness; if provided, used to create a RandomState.
      axis: Axis along which to sample. For a DataFrame with row sampling, use 'index' or 0.
    
    Returns:
      A DataFrame where:
        - If data is a Series: each column is a bootstrap sample.
        - If data is a DataFrame: the output columns are a MultiIndex where the outer level
          represents the bootstrap replicate (e.g., 'sample_1', 'sample_2', ...) and the inner
          level contains the original DataFrame's columns.

    For hypotheis testing:
    series_sample_1 = result_df['sample_1']['A']
    series_sample_2 = result_df['sample_2']['A']
    """
    rng = np.random.RandomState(random_state)
    samples = []
    
    for i in range(n):
        seed = rng.randint(0, 10**8)
        sample = data.sample(n=rows, frac=frac, replace=replace, weights=weights,
                             random_state=seed, axis=axis)
        if axis in (0, 'index'):
            # Always reset the index to ensure consistent concatenation.
            sample = sample.reset_index(drop=True)
        samples.append(sample)
    
    if isinstance(data, pd.DataFrame):
        # Concatenate along columns, using keys to create a MultiIndex.
        output = pd.concat(samples, axis=1, keys=[f'sample_{i+1}' for i in range(n)])
    else:
        # For Series, rename the columns to indicate the sample number.
        output = pd.concat(samples, axis=1)
        output.columns = [f'sample_{i+1}' for i in range(n)]
    
    return output

def downsample(
    df: pd.DataFrame,
    target: str = None,
    n_target_majority: Optional[int] = None,
    n_rows: Optional[int] = None,
    random_state: int = 12345,
) -> pd.DataFrame:
    """
    Downsample a DataFrame to address a class imbalance issue or to reduce overall size. 
    Optionally, rows with missing values in the target column can be dropped before processing.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing features and the target column.
        target (str): Name of the target column containing categorical labels (e.g., 0/1).
        n_target_majority (Optional[int]): The majority class will be downsampled to this total.
        n_rows (Optional[int]): Downsample majority to meet this overall size. 
        random_state (int): Random state for reproducibility.
        dropna (bool): If True, drop rows where the target column is NaN before processing.
    
    Returns:
        pd.DataFrame: A new DataFrame with the requested downsampling applied.
    
    """
    if n_target_majority is not None or n_rows is not None:
        # Identify majority (and implicitly minority) using value_counts.
        target_counts = df[target].value_counts()
        # Identify the majority label as the one with the highest count
        majority_label = target_counts.idxmax()
        
        # Split the DataFrame into majority and non-majority (minority) groups.
        df_majority = df[df[target] == majority_label]
        df_minority = df[df[target] != majority_label]
        
        # QC params
        if n_target_majority is not None and n_target_majority >= len(df_majority):
            raise ValueError(
                f"desired_majority ({n_target_majority}) is greater than the current majority count ({len(df_majority)})."
            )
        elif n_rows is not None and n_rows > len(df):
            raise ValueError(f'n_rows larger than df')
        else:
            pass

        #downsample
        if n_target_majority is not None:
            # downsample target majority
            df_majority = resample(
                df_majority,
                replace=False,
                n_samples=int(n_target_majority),
                random_state=random_state
            )
            # Recombine the groups (minority remains unchanged).
            df_downsampled = pd.concat([df_majority, df_minority]).reset_index(drop=True)
        
        elif n_rows is not None and n_target_majority is None:
            #downsample df
            df_downsampled = resample(
                    df,
                    replace=False,
                    n_samples=n_rows,
                    random_state=random_state
                )
        else:
            pass
            
        if n_target_majority is not None and n_rows is not None:
            #downsample df_downsampled to n_rows
            df_downsampled = resample(
                    df_downsampled,
                    replace=False,
                    n_samples=n_rows,
                    random_state=random_state
            )
        else:
            pass
        
        # Shuffle the final DataFrame to mix the rows.
        df_downsampled = shuffle(df_downsampled, random_state=random_state).reset_index(drop=True)
        
        print(f'df_downsampled shape: {df_downsampled.shape}')
        print(f'--- downsample() complete\n')
        return df_downsampled
    
    else:
        print(f'(no downsampling)')
        return df

def upsample(
    df: pd.DataFrame,
    target: str = None,
    n_target_minority: int = None,
    n_rows: int = None,
    random_state: int = 12345,
) -> pd.DataFrame:
    """
    Upsample a DataFrame for two possible reasons:
    
    1. To boost the minority class if it is too small.
    2. To enlarge the overall DataFrame if the total number of rows is too small.
    
    Optionally, it can drop rows with missing target values before processing.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing features and the target column.
        target (str): Name of the target column containing categorical labels (e.g., 0/1).
        desired_minority (Optional[int]): If provided and greater than the current minority count,
                                          the minority class will be upsampled to this total.
        desired_overall (Optional[int]): If provided and greater than the DataFrame's current size,
                                         the entire DataFrame will be upsampled (with replacement)
                                         to reach this overall number of rows.
        random_state (int): Random state for reproducibility.
        dropna (bool): If True, drop rows where the target column is NaN before processing.
    
    Returns:
        pd.DataFrame: A new DataFrame with the requested upsampling applied.
    
    Raises:
        ValueError: if desired_minority is lower than the current count of the minority class, or 
                    if desired_overall is lower than the current DataFrame size.
    """
    
    # Identify minority and majority classes from remaining rows
    target_counts = df[target].value_counts()
    minority_label = target_counts.idxmin()
    majority_label = target_counts.idxmax()
    
    df_minority = df[df[target] == minority_label]
    df_majority = df[df[target] == majority_label]
    
    # Upsample the minority class if desired number is provided and is larger than current count
    if n_target_minority is not None:
        if n_target_minority < len(df_minority):
            raise ValueError(
                f"desired_minority ({n_target_minority}) is less than the current minority count ({len(df_minority)})."
            )
        df_minority = resample(
            df_minority,
            replace=True,
            n_samples=int(n_target_minority),
            random_state=random_state
        )
    
    # Recombine the classes after upsampling minority if needed
    df_upsampled = pd.concat([df_majority, df_minority]).reset_index(drop=True)
    
    # Upsample the overall DataFrame if n_rows is provided
    if n_rows is not None:
        if n_rows < len(df_upsampled):
            raise ValueError(
                f"desired_overall ({n_rows}) is less than or equal to the current total rows ({len(df_upsampled)})."
            )
        else:
            df_upsampled = resample(
            df_upsampled,
            replace=True,
            n_samples=n_rows,
            random_state=random_state
        )
    
    if n_target_minority is not None or n_rows is not None:
        # One final shuffle to mix any duplicated entries
        df_upsampled = shuffle(df_upsampled, random_state=random_state).reset_index(drop=True)
        
        print(f'df_upsampled shape: {df_upsampled.shape}\n')
        print(f'-- upsample() complete\n')
        return df_upsampled
    
    else:

        print(f'(no upsampling)')
    return df_upsampled

def ordinal_encoder(df, ordinal_cols):
    if ordinal_cols is not None:
        encoded_values_dict = []
        for col in ordinal_cols:
            print(f'Encoding column: {col}')
            
            # Create a mapping dictionary: each unique value to an integer based on its position
            unique_values = sorted(df[col].dropna().unique())
            mapping_dict = {val: idx for idx, val in enumerate(unique_values)}
            print(f'Mapping values {col}: {mapping_dict}')
            df[col] = df[col].map(mapping_dict)
            
            # Store the mapping for potential later use
            encoded_values_dict.append(mapping_dict)
        
        print(f'ordinal_encoder() complete\n')
        return df, encoded_values_dict
    else:
        return df, None

def missing_values(
        df: pd.DataFrame,
        missing_values_method: str,
        fill_value=0) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame based on the specified method.
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    method (str): 'drop', 'fill', 'mean', 'median', or 'mode'.
    fill_value: the value to use with the 'fill' method. Can be any type (text, number, etc.). Defaults to 0.
                    
    Returns:
        pd.DataFrame: The DataFrame after handling missing values.
    """
    if missing_values_method == 'drop':
        df = df.dropna()
    elif missing_values_method == 'fill':
        df = df.fillna(fill_value)
    elif missing_values_method == 'mean':
        df = df.fillna(df.mean())
    elif missing_values_method == 'median':
        df = df.fillna(df.median())
    elif missing_values_method == 'mode':
        df = df.fillna(df.mode().iloc[0])
    elif missing_values_method is None:
        print(f'(no missing values method applied)')
        return df
    else:
        raise ValueError(f"Unknown method: {missing_values_method}")
    
    print(f'df shape: {df.shape}')
    print(f'--- missing_values() complete\n')
    return df

def feature_scaler(df):
    # feature scaling
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[col])

    print(f'--- feature_scaler() complete\n')
    return df

def categorical_encoder(df, model_type):
    if model_type == 'Regressions':
        # dummy vars (one-hot encoding) for categorial vars
        df_ohe = pd.get_dummies(df, drop_first=True)
        df = df_ohe

        print(f'-- categorical_encoder() complete\n')
        return df
    elif model_type == 'Machine Learning':
        # Label Encoding for categorical vars
        encoder = OrdinalEncoder()
        encoder.fit_transform(df)
        df_ordinal = pd.DataFrame(encoder.transform(df), columns=df.columns)
        df = df_ordinal

        print(f'--- categorical_encoder() complete\n')
        return df
    
    else:
        print(f'(no categorical encoder applied)')
        return df
    
def data_splitter(
    df: pd.DataFrame,
    split_ratio: tuple = (),
    target: str = None,
    random_state: int = None,
) -> tuple:
    """
    Splits a DataFrame into training, validation, and optionally test sets based on the provided split ratios.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        split_ratio (tuple): If two values (train_ratio, validation_ratio), if three values (train_ratio, validation_ratio, test_ratio).
        target (str): The column name to be used as the target variable.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: 
            - For two ratios: (train_features, train_target, valid_features, valid_target)
            - For three ratios: (train_features, train_target, valid_features, valid_target, test_features, test_target)
    """
    print(f'Running data_splitter()...')
    print(f'df shape start: {df.shape}')
    if len(split_ratio) == 0 or split_ratio is None:
        print(f'(no splitting)')
        return df
    
    elif len(split_ratio) == 1:
        if split_ratio <= 1:
            df = downsample(df)

            print(f'--- data_splitter() complete\n')
            return df
        
        elif split_ratio > 1:
            df = upsample(df)

            print(f'--- data_splitter() complete\n')
            return df

    elif len(split_ratio) == 2:
        train_ratio, val_ratio = split_ratio

        # Split data into training and validation sets.
        df_train, df_valid = train_test_split(df, test_size=val_ratio, random_state=random_state)

        print(f"Shapes:\ndf_train: {df_train.shape}\ndf_valid: {df_valid.shape}")

        train_features = df_train.drop(target, axis=1)
        train_target = df_train[target]
        valid_features = df_valid.drop(target, axis=1)
        valid_target = df_valid[target]

        print(f'--- data_splitter() complete\n')
        return train_features, train_target, valid_features, valid_target

    elif len(split_ratio) == 3:
        train_ratio, val_ratio, test_ratio = split_ratio

        # First split: separate out the test set.
        df_temp, df_test = train_test_split(df, test_size=test_ratio, random_state=random_state)

        # Recalculate validation ratio relative to the remaining data (df_temp).
        new_val_ratio = val_ratio / (1 - test_ratio)

        df_train, df_valid = train_test_split(df_temp, test_size=new_val_ratio, random_state=random_state)

        print(f"new data shapes:\ndf_train: {df_train.shape}\ndf_valid: {df_valid.shape}\ndf_test: {df_test.shape}")

        train_features = df_train.drop(target, axis=1)
        train_target = df_train[target]
        valid_features = df_valid.drop(target, axis=1)
        valid_target = df_valid[target]
        test_features = df_test.drop(target, axis=1)
        test_target = df_test[target]

        print(f'--- data_splitter() complete\n')
        return train_features, train_target, valid_features, valid_target, test_features, test_target

    else:
        raise ValueError("split_ratio must be a tuple with 3 or fewer elements.")

def data_transformer(
        df: pd.DataFrame,
        split_ratio: tuple = (),
        target: str = None,
        n_target_majority: int = None,
        n_target_minority: int = None,
        n_rows: int = None,
        ordinal_cols: list = None,
        missing_values_method: str = None,
        fill_value: any = None,
        random_state: int = None,
        model_type: str = None,
        scale_features: bool = True
    ): 
    """
    data splitter, encoding, based on desired data for modeling.   
    
    Parameters:
        df: pd.DataFrame, The input dataframe
        n_rows: int = None, desired number of rows
        split_ratio: tuple = None, ratio of train, validate and test sets. eg (.75, .25) or (.6, .2, .2)
        target: str = None, Name of the target column.
        n_target: int = None, desired number of target rows.
        ordinal_cols: list = None, List of ordinal variable names.
        missing_values: str = None,
        random_state: int = None, Random state for reproducibility.
        model: str = None, model to be used on output data sets.
        feature_scale: bool = None, scale continuous variables if True.
        
    Returns:
        If two splits: train_features, train_target, valid_features, valid_target.
        If three splits: train_features, train_target, valid_features, valid_target, test_features, test_target.
    """
    print(f'Running data_transformer()...')
    #print(f'df shape start: {df.shape}')
    # QC parameters
    if split_ratio is None:
        split_ratio = ()
    
    # Validate the length of split_ratio and that the values sum to 1
    if (len(split_ratio) not in [0, 1, 2, 3]):
        raise ValueError("split_ratio must be a tuple of length 0 (pass), 1 (resample), 2 (train, validate) or 3 (train, validate, test).")
    
    if len(split_ratio) > 2:
        if not abs(sum(split_ratio) - 1.0) < 1e-6:
            raise ValueError("The elements of split_ratio must sum to 1.")

    #Downsample if applicable
    #for when downsampling params are provided.  it's done again if split ratio < 1.  
    df = downsample(
        df,
        target,
        n_target_majority,
        n_rows,
        random_state
    )    

    # Upsample if params provided.  duplicated again if split ratio > 1. 
    df = upsample(
        df,
        target,
        n_target_minority,
        n_rows,
        random_state,
    )

    # apply up/down sample first:
    if len(split_ratio) == 1:
            df = data_splitter(df, split_ratio)
            return df
    else:
        pass
    
    # handling missing data
    df = missing_values(df, missing_values_method, fill_value)

    # Encode ordinal columns if specified
    df, encoded_values_dict = ordinal_encoder(df, ordinal_cols)

    # Encode categorial columns: one-hot encoding for regression (regression), Label Encoding for ML
    df = categorical_encoder(df, model_type)            

    # feature scaling for regression models
    if scale_features is False:
        pass
    elif scale_features is True and model_type == 'Regressions' or None:
        df = feature_scaler(df)
    else:
        pass
    

    # Split data
    try:
        if len(split_ratio) == 0:
            print(f'df count: 1')
            return df

        if len(split_ratio) == 2:
            train_features, train_target, valid_features, valid_target = data_splitter(
                df,
                split_ratio,
                target,
                random_state,
            )
            print(f'df count: 4')
            return train_features, train_target, valid_features, valid_target
        
        elif len(split_ratio) == 3:
            train_features, train_target, valid_features, valid_target, test_features, test_target = data_splitter(
                df,
                split_ratio,
                target,
                random_state,
            )
            print(f'df count: 6')
            return train_features, train_target, valid_features, valid_target, test_features, test_target
        
    except Exception as e:
        print(f"(no splitting): {e}\n")
