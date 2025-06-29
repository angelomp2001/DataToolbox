# from data_transformer import downsample, upsample, ordinal_encoder, missing_values, feature_scaler, categorical_encoder, data_splitter, data_transformer

# input df and requirements, outputs processed df
# processing: downsample, upsample, ordinal_encoder, missing_values, feature_scaler, categorical_encoder, data_splitter, data_transformer

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, shuffle
from typing import Optional

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
        
    # Identify majority (and implicitly minority) using value_counts.
    target_counts = df[target].value_counts()
    # Identify the majority label as the one with the highest count
    majority_label = target_counts.idxmax()
    
    # Split the DataFrame into majority and non-majority (minority) groups.
    df_majority = df[df[target] == majority_label]
    df_minority = df[df[target] != majority_label]
    
    # Downsample the majority class if desired_majority is provided.
    if n_target_majority is not None:
        if n_target_majority > len(df_majority):
            raise ValueError(
                f"desired_majority ({n_target_majority}) is greater than the current majority count ({len(df_majority)})."
            )
        
        # If equal then no need to downsample
        if n_target_majority < len(df_majority):
            df_majority = resample(
                df_majority,
                replace=False,
                n_samples=n_target_majority,
                random_state=random_state
            )
    
    # Recombine the groups (minority remains unchanged).
    df_downsampled = pd.concat([df_majority, df_minority]).reset_index(drop=True)
    
    # Downsample overall DataFrame if desired_overall is provided.
    if n_rows is not None:
        if n_rows > len(df_downsampled):
            raise ValueError(
                f"desired_overall ({n_rows}) is greater than the current total rows ({len(df_downsampled)})."
            )
        # Downsample the entire DataFrame while roughly maintaining the class proportions.
        df_downsampled = resample(
            df_downsampled,
            replace=False,
            n_samples=n_rows,
            random_state=random_state
        )
    
    # Shuffle the final DataFrame to mix the rows.
    df_downsampled = shuffle(df_downsampled, random_state=random_state).reset_index(drop=True)
    
    print(f'-- downsample() complete\n')
    return df_downsampled

def upsample(
    df: pd.DataFrame,
    target: str = None,
    desired_minority: Optional[int] = None,
    desired_overall: Optional[int] = None,
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
    if desired_minority is not None:
        if desired_minority < len(df_minority):
            raise ValueError(
                f"desired_minority ({desired_minority}) is less than the current minority count ({len(df_minority)})."
            )
        df_minority = resample(
            df_minority,
            replace=True,
            n_samples=desired_minority,
            random_state=random_state
        )
    
    # Recombine the classes after upsampling minority if needed
    df_upsampled = pd.concat([df_majority, df_minority]).reset_index(drop=True)
    
    # Upsample the overall DataFrame if desired_overall is provided
    if desired_overall is not None:
        if desired_overall < len(df_upsampled):
            raise ValueError(
                f"desired_overall ({desired_overall}) is less than the current total rows ({len(df_upsampled)})."
            )
        df_upsampled = resample(
            df_upsampled,
            replace=True,
            n_samples=desired_overall,
            random_state=random_state
        )
    
    # One final shuffle to mix any duplicated entries
    df_upsampled = shuffle(df_upsampled, random_state=random_state).reset_index(drop=True)
    
    print(f'-- upsample() complete\n')
    return df_upsampled

def ordinal_encoder(df, ordinal_cols):
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
    if missing_values_method == 'drop' or missing_values_method is None:
        df = df.dropna()
    elif missing_values_method == 'fill':
        df = df.fillna(fill_value)
    elif missing_values_method == 'mean':
        df = df.fillna(df.mean())
    elif missing_values_method == 'median':
        df = df.fillna(df.median())
    elif missing_values_method == 'mode':
        df = df.fillna(df.mode().iloc[0])
    else:
        raise ValueError(f"Unknown method: {missing_values_method}")
    
    print(f'--- missing_values() complete\n')
    return df

def feature_scaler(df):
    # feature scaling
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])

    print(f'--- feature_scaler() complete\n')
    return df

def categorical_encoder(df, model_name):
    if model_name == 'LogisticRegression' or None:
        # dummy vars (one-hot encoding) for categorial vars
        df_ohe = pd.get_dummies(df, drop_first=True)
        df = df_ohe

        print(f'-- categorical_encoder() complete\n')
        return df
    elif model_name == 'DecisionTreeClassifier' or 'RandomForestClassifier':
        # Label Encoding for categorical vars
        encoder = OrdinalEncoder()
        encoder.fit_transform(df)
        df_ordinal = pd.DataFrame(encoder.transform(df), columns=df.columns)
        df = df_ordinal

        print(f'--- categorical_encoder() complete\n')
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
    if len(split_ratio) == 0 or split_ratio is None:

        print(f'--- data_splitter() complete\n')
        return df
    
    if len(split_ratio) == 1:
        if split_ratio <= 1:
            df = downsample(df)

            print(f'--- data_splitter() complete\n')
            return df
        
        elif split_ratio > 1:
            df = upsample(df)

            print(f'--- data_splitter() complete\n')
            return df

    if len(split_ratio) == 2:
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
        n_rows: int = None,
        split_ratio: tuple = (),
        target: str = None,
        n_target_majority: Optional[int] = None,
        ordinal_cols: list = None,
        missing_values_method: str = None,
        fill_value: any = None,
        random_state: int = None,
        model_name: str = None,
        feature_scaler: bool = None

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
    # QC parameters
    # Validate the length of split_ratio and that the values sum to 1
    if len(split_ratio) not in [0, 1, 2, 3]:
        raise ValueError("split_ratio must be a tuple of length 0 (pass), 1 (resample), 2 (train, validate) or 3 (train, validate, test).")
    
    if len(split_ratio) > 2:
        if not abs(sum(split_ratio) - 1.0) < 1e-6:
            raise ValueError("The elements of split_ratio must sum to 1.")

    # Downsample if applicable
    try:
        print('-- downsampling...')
        df = downsample(
            df,
            target,
            n_target_majority,
            n_rows,
            random_state,
            missing_values_method,
        )    
    except Exception as e:
        print(f"(no downsampling): {e}\n")

    # Upsample if applicable
    try:
        print('-- upsampling...')
        df = upsample(
            df,
            target,
            n_target_majority,
            n_rows,
            random_state,
            missing_values_method,
        )
    except Exception as e:
        print(f"(no upsampling): {e}\n")

    # handling missing data
    try:
        print('-- addressing missing values...')
        df = missing_values(df, missing_values_method, fill_value)
    except Exception as e:
        print(f"(no missing values): {e}\n")

    # Encode ordinal columns if specified
    try:
        print('-- ordinal encoding...')
        df, encoded_values_dict = ordinal_encoder(df, ordinal_cols)

    except Exception as e:
        print(f"(no ordinal vars to encode): {e}\n")

    # Encode categorial columns: one-hot encoding for regression (regression), Label Encoding for ML
    try:
        print('-- categorical encoding...')
        df = categorical_encoder(df, model_name)            
    except Exception as e:
        print(f"(no categorical vars to encode): {e}\n")

    # feature scaling for regression models
    try:
        if model_name == 'LogisticRegression' or None:
            print('-- feature scaling...')
            df = feature_scaler(df)
        else:
            pass
    except Exception as e:
        print(f'(no feature scaling): {e}\n')
    

    # Split data
    try:
        print('-- splitting data...')
        if len(split_ratio) == 0:
            
            print(f'data_transformer() complete\n')
            return df
        
        if len(split_ratio) == 1:
            df = data_splitter(df, split_ratio)
            
            print(f'data_transformer() complete\n')
            return df

        if len(split_ratio) == 2:
            train_features, train_target, valid_features, valid_target = data_splitter(
                df,
                split_ratio,
                target,
                random_state,
            )
            print(f'data_transformer() complete\n')
            return train_features, train_target, valid_features, valid_target
        
        elif len(split_ratio) == 3:
            train_features, train_target, valid_features, valid_target, test_features, test_target = data_splitter(
                df,
                split_ratio,
                target,
                random_state,
            )
            print(f'data_transformer() complete\n')
            return train_features, train_target, valid_features, valid_target, test_features, test_target
        
    except Exception as e:
        print(f"(no splitting): {e}\n")