# instructions for recording functions into a class
# Create a new class
# Copy each function into the class as a method

# In __init__: List common parameters (df) used in any of the related functions
# - Assign these parameters to instance variables (self.xxx)

# per function:
# - add self to function
# - remove default value from self parameters
# - update function parameters that are now instance variables with self.parameter_name
# test method

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, shuffle
from typing import Optional
import numpy as np
from typing import Union



class DataProcessor:
    def __init__(
        self,
        df: pd.DataFrame,
        random_state: int = 12345,
        target: str = None,
    ):
        self.df = df
        self.random_state = random_state
        self.target = target
        self.features = self.df.drop(columns=[target], axis=1) if target else None
        self.train_features = None
        self.train_target = None
        self.valid_features = None
        self.valid_target = None
        self.test_features = None
        self.test_target = None
        self.w = None
        self.w0 = None





    def bootstrap(self,
        n: int = 100,
        n_rows: int = None,
        frac_rows: float = None,
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
        if random_state is None:
            random_state = self.random_state
        else:
            pass

        rng = np.random.RandomState(random_state)
        samples = []
        
        for i in range(n):
            seed = rng.randint(0, 10**8)
            sample = df.sample(n=n, frac=frac_rows, replace=replace, weights=weights,
                                random_state=seed, axis=axis)
            if axis in (0, 'index'):
                # Always reset the index to ensure consistent concatenation.
                sample = sample.reset_index(drop=True)
            samples.append(sample)
        
        if isinstance(self.df, pd.DataFrame):
            # Concatenate along columns, using keys to create a MultiIndex.
            new_df = pd.concat(samples, axis=1, keys=[f'sample_{i+1}' for i in range(n)])
        else:
            # For Series, rename the columns to indicate the sample number.
            new_df = pd.concat(samples, axis=1)
            new_df.columns = [f'sample_{i+1}' for i in range(n)]
        
        return new_df
                 
    def downsample(
    self,
    target: str = None,
    n_target_majority: Optional[int] = None,
    n_rows: Optional[int] = None,
    random_state: int = None,
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
        if random_state is None:
            random_state = self.random_state
        else:
            pass

        if n_target_majority is not None or n_rows is not None:
            # Identify majority (and implicitly minority) using value_counts.
            target_counts = self.df[target].value_counts()
            # Identify the majority label as the one with the highest count
            majority_label = target_counts.idxmax()
            
            # Split the DataFrame into majority and non-majority (minority) groups.
            df_majority = self.df[self.df[target] == majority_label]
            df_minority = self.df[self.df[target] != majority_label]
            
            # QC params
            if n_target_majority is not None and n_target_majority >= len(df_majority):
                raise ValueError(
                    f"desired_majority ({n_target_majority}) is greater than the current majority count ({len(df_majority)})."
                )
            elif n_rows is not None and n_rows > len(self.df):
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
                        self.df,
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
            return self.df

    def upsample(self,
    target: str = None,
    n_target_minority: int = None,
    n_rows: int = None,
    random_state: int = None,
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
        if random_state is None:
            random_state = self.random_state
        else:
            pass

        # Identify minority and majority classes from remaining rows
        target_counts = self.df[target].value_counts()
        minority_label = target_counts.idxmin()
        majority_label = target_counts.idxmax()
        
        df_minority = self.df[self.df[target] == minority_label]
        df_majority = self.df[self.df[target] == majority_label]
        
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
 
    def missing_values(
        self,
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
            self.df = self.df.dropna()
        elif missing_values_method == 'fill':
            self.df = self.df.fillna(fill_value)
        elif missing_values_method == 'mean':
            self.df = self.df.fillna(self.df.mean())
        elif missing_values_method == 'median':
            self.df = self.df.fillna(self.df.median())
        elif missing_values_method == 'mode':
            self.df = self.df.fillna(self.df.mode().iloc[0])
        elif missing_values_method is None:
            print(f'(no missing values method applied)')
            return self.df
        else:
            raise ValueError(f"Unknown method: {missing_values_method}")
        
        print(f'df shape: {self.df.shape}')
        print(f'--- missing_values() complete\n')
        return self.df
    
    def feature_scaler(self) -> pd.DataFrame:
        '''
        Scales all numeric features in self.df using StandardScaler.
        input: pandas DataFrames or numpy arrays/matrices.
        output: scaled DataFrame or ndarray.
        '''
        if isinstance(self.df, (np.ndarray, np.matrix)):
            scaler = StandardScaler()
            scaled = scaler.fit_transform(self.df)
            print(f'--- feature_scaler() complete (array input)\n')
            return scaled

        # If self.df is a DataFrame
        scaled_data = self.df.copy()
        for col in scaled_data.select_dtypes(include=[np.number]).columns:
            scaler = StandardScaler()

            # keeps it as 2D DataFrame
            scaled_data[col] = scaler.fit_transform(
                scaled_data[[col]]  
            ).flatten()

        print(f'--- feature_scaler() complete (DataFrame input)\n')
        return scaled_data

    def encode_features(
    self,
    model_type: str,
    ordinal_cols: Union[list, dict] = None,
    categorical_cols: list = None,
    auto_encode: bool = False
    ) -> pd.DataFrame:
        """
        Encodes features for modeling:
        - For 'Regressions': One-hot encodes categorical_cols.
        - For 'Machine Learning': Ordinal encodes ordinal_cols and categorical_cols.
        Ensures no column remains with dtype 'object' or 'string'.

        Parameters:
            model_type (str): 'Regressions' or 'Machine Learning'
            ordinal_cols (list or dict): columns to be ordinal encoded.
                                        If dict, values must be ordered list (low → high)
            categorical_cols (list): columns to be one-hot or ordinal encoded
            auto_encode (bool): infer object columns if no lists are provided

        Returns:
            self.df (pd.DataFrame): the fully encoded DataFrame
            encoded_values_dict (dict): mappings used for ordinal and categorical encoding
        """


        encoded_values_dict = {'ordinal': {}, 'categorical': {}}
        object_cols = self.df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

        # Prepare encoding targets
        if auto_encode:
            if ordinal_cols is None:
                ordinal_cols = []
            if categorical_cols is None:
                categorical_cols = object_cols
            ordinal_cols = list(set(ordinal_cols)) if isinstance(ordinal_cols, list) else ordinal_cols
            categorical_cols = list(set(categorical_cols) - set(ordinal_cols if isinstance(ordinal_cols, list) else ordinal_cols.keys()))

        # Handle ordinal encoding
        if ordinal_cols:
            if isinstance(ordinal_cols, dict):
                for col, order in ordinal_cols.items():
                    print(f"Ordinal encoding column (custom order): {col}")
                    mapping_dict = {val: idx for idx, val in enumerate(order)}
                    self.df[col] = self.df[col].map(mapping_dict)
                    encoded_values_dict['ordinal'][col] = mapping_dict

            elif isinstance(ordinal_cols, list):
                for col in ordinal_cols:
                    print(f"Ordinal encoding column (auto): {col}")
                    col_data = self.df[col]

                    # Try datetime conversion if dtype is object/string
                    if col_data.dtype in ['object', 'string']:
                        try:
                            col_data = pd.to_datetime(col_data)
                            self.df[col] = col_data  # Update with datetime if successful
                        except Exception:
                            pass

                    unique_values = sorted(col_data.dropna().unique())
                    mapping_dict = {val: idx for idx, val in enumerate(unique_values)}
                    self.df[col] = self.df[col].map(mapping_dict)
                    encoded_values_dict['ordinal'][col] = mapping_dict

        # Handle categorical encoding
        if model_type == 'Regressions' and categorical_cols:
            print(f'One-hot encoding columns: {categorical_cols}')
            self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)

        elif model_type == 'Machine Learning' and categorical_cols:
            for col in categorical_cols:
                if self.df[col].dtype in ['object', 'string', 'category']:
                    print(f"Ordinal encoding column (categorical): {col}")
                    unique_values = sorted(self.df[col].dropna().unique())
                    mapping_dict = {val: idx for idx, val in enumerate(unique_values)}
                    self.df[col] = self.df[col].map(mapping_dict)
                    encoded_values_dict['categorical'][col] = mapping_dict

        # Final check: ensure no object columns remain
        str_cols_remaining = self.df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        if str_cols_remaining:
            raise ValueError(f"The following columns still have object dtype: {str_cols_remaining}")

        print('-- Encoding complete. No string columns remain.\n')
        return self.df, encoded_values_dict if encoded_values_dict['ordinal'] or encoded_values_dict['categorical'] else None

    def split(
    self,
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
        # save relevant new inputs to self. 
        # string
        self.target = target

        # df
        self.features = self.df.drop(self.target, axis=1)

        if random_state is None:
            random_state = self.random_state
        else:
            pass

        print(f'Running data_splitter()...')
        print(f'df shape start: {self.df.shape}')
        if len(split_ratio) == 0 or split_ratio is None:
            print(f'(no splitting)')
            return self.df
        
        elif len(split_ratio) == 1:
            if split_ratio <= 1:
                self.df = self.downsample(self.df, n_rows=split_ratio*len(self.df))

                print(f'--- data_splitter() complete\n')
                return self.df
            
            elif split_ratio > 1:
                self.df = self.upsample(self.df, n_rows=split_ratio*len(self.df))

                print(f'--- data_splitter() complete\n')
                return self.df

        elif len(split_ratio) == 2:
            train_ratio, val_ratio = split_ratio

            # Split data into training and validation sets.
            df_train, df_valid = train_test_split(self.df, test_size=val_ratio, random_state=random_state)

            print(f"Shapes:\ndf_train: {df_train.shape}\ndf_valid: {df_valid.shape}")

            self.train_features = df_train.drop(self.target, axis=1)
            self.train_target = df_train[self.target]
            self.valid_features = df_valid.drop(self.target, axis=1)
            self.valid_target = df_valid[self.target]

            print(f'--- data_splitter() complete\n')
            return self.train_features, self.train_target, self.valid_features, self.valid_target

        elif len(split_ratio) == 3:
            train_ratio, val_ratio, test_ratio = split_ratio

            # First split: separate out the test set.
            df_temp, self.df_test = train_test_split(self.df, test_size=test_ratio, random_state=random_state)

            # Recalculate validation ratio relative to the remaining data (df_temp).
            new_val_ratio = val_ratio / (1 - test_ratio)

            self.df_train, self.df_valid = train_test_split(df_temp, test_size=new_val_ratio, random_state=random_state)

            print(f"new data shapes:\ndf_train: {self.df_train.shape}\ndf_valid: {self.df_valid.shape}\ndf_test: {self.df_test.shape}")

            self.train_features = self.df_train.drop(self.target, axis=1)
            self.train_target = self.df_train[self.target]
            self.valid_features = self.df_valid.drop(self.target, axis=1)
            self.valid_target = self.df_valid[self.target]
            self.test_features = self.df_test.drop(self.target, axis=1)
            self.test_target = self.df_test[self.target]

            print(f'--- data_splitter() complete\n')
            return self.train_features, self.train_target, self.valid_features, self.valid_target, self.test_features, self.test_target

        else:
            raise ValueError("split_ratio must be a tuple with 3 or fewer elements.")

    def vectorize(
            self,
            df: pd.DataFrame = None,
            features: pd.DataFrame or pd.Series= None,
            target: str = None,
        ) -> pd.DataFrame:
        '''
        Vectorizes df
        '''
        # vectorize:
        if df is not None:
            self.df = df
            return self.df
        if features is not None and target is not None:
            self.features = features.to_numpy() if isinstance(features, pd.DataFrame) else features
            self.target = target.to_numpy() if isinstance(target, (pd.Series, pd.DataFrame)) else target
            return self.features, self.target

        if df is None and features is None and target is None:
            self.train_features = self.train_features.to_numpy() if self.train_features is not None else None
            self.train_target = self.train_target.to_numpy() if self.train_target is not None else None
            self.valid_features = self.valid_features.to_numpy() if self.valid_features is not None else None
            self.valid_target = self.valid_target.to_numpy() if self.valid_target is not None else None
            self.test_features = self.test_features.to_numpy() if self.test_features is not None else None
            self.test_target = self.test_target.to_numpy() if self.test_target is not None else None
            self.features = self.features.to_numpy() if self.features is not None else None
            self.target = self.target.to_numpy() if self.target is not None else None
            self.df = np.concatenate((self.features, self.target.reshape(-1, 1)), axis=1) if self.features is not None and self.target is not None else None   
            return self.train_features, self.train_target, self.valid_features, self.valid_target, self.test_features, self.test_target, self.features, self.target, self.df






# df = pd.DataFrame({
#     'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
#     'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
#     'age': [25, 30, 35, 40, 45],
#     'salary': [50000, 60000, 75000, 80000, 90000],
#     'department': ['HR', 'IT', 'Sales', 'Marketing', 'Finance']
# })
# data = DataProcessor(df)

# data.df, encoded_values_dict = data.encode_features(
#     model_type='Machine Learning',
#     categorical_cols=['department', 'name'],
#     ordinal_cols=['date']
# )
# output = data.df
# print(encoded_values_dict)

