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
        target_name: str = None,
    ):
        assert isinstance(df, pd.DataFrame), "`df` must be a pandas DataFrame."
        self.df = df
        self.random_state = random_state
        self.target_name = target_name
        self.target = df[target_name] if target_name else None
        self.features = self.df.drop(columns=[target_name], axis=1) if target_name else None
        self.train_features = None
        self.train_target = None
        self.valid_features = None
        self.valid_target = None
        self.test_features = None
        self.test_target = None
        self.w = None
        self.w0 = None
        self.train_features_vectorized = None
        self.train_target_vectorized = None
        self.valid_features_vectorized = None
        self.valid_target_vectorized = None
        self.test_features_vectorized = None
        self.test_target_vectorized = None
        self.features_vectorized = None
        self.target_vectorized = None
        self.df_vectorized = None
        self.encoded_values_dict = {'ordinal': {}, 'categorical': {}}




    def bootstrap(self,
        n: int = 100,
        n_rows: int = None,
        frac_rows: float = None,
        replace: bool = True,
        weights: Optional[np.ndarray] = None,
        random_state: int = None,
        axis: str = 'index',
        show: bool = False
    ):
        """
    Create n bootstrap samples from the data, resetting index each time.

    Parameters:
    - n: Number of bootstrap samples.
    - n_rows: Number of rows per sample. If given, 'frac_rows' is ignored.
    - frac_rows: Fraction of rows per sample, used only if n_rows is None.
    - replace: Sample with replacement (default True).
    - weights: Sampling weights; default is uniform.
    - random_state: Seed for reproducibility.
    - axis: Sampling axis ('index' or 0 for rows).
    - show: Print first few rows of result.

    Returns:
    - self, with self.df as a MultiIndex DataFrame of samples.
    
    Example:
        dp.bootstrap(n=10).df['sample_1']['A']
    """
        if random_state is None:
            random_state = self.random_state
        else:
            pass

        rng = np.random.RandomState(random_state)
        samples = []
        
        for i in range(n):
            seed = rng.randint(0, 10**8)
            sample = self.df.sample(
                n=n_rows if n_rows is not None else None,
                frac=frac_rows if n_rows is None else None,
                replace=replace,
                weights=weights,
                random_state=seed,
                axis=axis
            )

            if axis in (0, 'index'):
                # Always reset the index to ensure consistent concatenation.
                sample = sample.reset_index(drop=True)
            samples.append(sample)
        
        if isinstance(self.df, pd.DataFrame):
            # Concatenate along columns, using keys to create a MultiIndex.
            self.df = pd.concat(samples, axis=1, keys=[f'sample_{i+1}' for i in range(n)])
        else:
            # For Series, rename the columns to indicate the sample number.
            self.df = pd.concat(samples, axis=1)
            self.df.columns = [f'sample_{i+1}' for i in range(n)]
        
        if show:
            print(self.df.head(15))

        return self
                 
    def downsample(
    self,
    target_name: str = None,
    n_target_majority: Optional[int] = None,
    n_rows: Optional[int] = None,
    random_state: int = None,
    show: bool = False,
    ) -> pd.DataFrame:
        """
        Downsample a DataFrame to address a class imbalance issue or to reduce overall size. 
        Optionally, rows with missing values in the target column can be dropped before processing.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing features and the target column.
            target_name (str): Name of the target column containing categorical labels (e.g., 0/1).
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
            target_counts = self.df[target_name].value_counts()
            # Identify the majority label as the one with the highest count
            majority_label = target_counts.idxmax()
            
            # Split the DataFrame into majority and non-majority (minority) groups.
            df_majority = self.df[self.df[target_name] == majority_label]
            df_minority = self.df[self.df[target_name] != majority_label]
            
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
                self.df = pd.concat([df_majority, df_minority]).reset_index(drop=True)
            
            elif n_rows is not None and n_target_majority is None:
                #downsample df
                self.df = resample(
                        self.df,
                        replace=False,
                        n_samples=n_rows,
                        random_state=random_state
                    )
            else:
                pass
                
            if n_target_majority is not None and n_rows is not None:
                #downsample df_downsampled to n_rows
                self.df = resample(
                        self.df,
                        replace=False,
                        n_samples=n_rows,
                        random_state=random_state
                )
            else:
                pass
            
            # Shuffle the final DataFrame to mix the rows.
            self.df = shuffle(self.df, random_state=random_state).reset_index(drop=True)
            
            print(f'df_downsampled shape: {self.df.shape}')
            print(f'--- downsample() complete\n')

            if show:
                print(self.df.head(15))
            return self
        
        else:
            print(f'(no downsampling)')
            return self

    def upsample(self,
        target_name: str = None,
        n_target_minority: int = None,
        n_rows: int = None,
        random_state: int = None,
        show: bool = False
        ) -> pd.DataFrame:
        """
        Upsample a DataFrame for two possible reasons:
        
        1. To boost the minority class if it is too small.
        2. To enlarge the overall DataFrame if the total number of rows is too small.
        
        Optionally, it can drop rows with missing target values before processing.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing features and the target column.
            target_name (str): Name of the target column containing categorical labels (e.g., 0/1).
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
        target_counts = self.df[target_name].value_counts()
        minority_label = target_counts.idxmin()
        majority_label = target_counts.idxmax()
        
        df_minority = self.df[self.df[target_name] == minority_label]
        df_majority = self.df[self.df[target_name] == majority_label]
        
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
        self.df = pd.concat([df_majority, df_minority]).reset_index(drop=True)
        
        # Upsample the overall DataFrame if n_rows is provided
        if n_rows is not None:
            if n_rows < len(self.df):
                raise ValueError(
                    f"desired_overall ({n_rows}) is less than or equal to the current total rows ({len(self.df)})."
                )
            else:
                self.df = resample(
                self.df,
                replace=True,
                n_samples=n_rows,
                random_state=random_state
            )
        
        if n_target_minority is not None or n_rows is not None:
            # One final shuffle to mix any duplicated entries
            self.df = shuffle(self.df, random_state=random_state).reset_index(drop=True)
            
            print(f'df_upsampled shape: {self.df.shape}\n')
            print(f'-- upsample() complete\n')

            if show:
                print(self.df.head(15))
            return self
        
        else:

            print(f'(no upsampling)')
        return self
 
    def missing_values(
        self,
        missing_values_method: str,
        fill_value=0,
        show: bool = False
    ) -> pd.DataFrame:
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
            self.df = self.df.dropna().copy()
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
            return self
        else:
            raise ValueError(f"Unknown method: {missing_values_method}")
        
        if show:
            print(self.df.head(15))
        print(f'df shape: {self.df.shape}')
        print(f'--- missing_values() complete\n')
        return self
    
    def feature_scaler(
            self,
            column_names: list = None,
            show: bool = False) -> pd.DataFrame:
        '''
        Scales all numeric features in self.df using StandardScaler.
        input: pandas DataFrames or numpy arrays/matrices.
        output: scaled DataFrame or ndarray.
        '''
        if isinstance(self.df, (np.ndarray, np.matrix)):
            scaler = StandardScaler()
            self.df = scaler.fit_transform(self.df)
            print(f'--- feature_scaler() complete (array input)\n')
            return self

        # If self.df is a DataFrame
        scaled_data = self.df.copy()

        if column_names is None:
            # If no specific columns are provided, scale all numeric columns
            column_names = scaled_data.select_dtypes(include=[np.number]).columns.tolist()

        # Check if the specified columns are numeric
        numeric_columns = scaled_data.select_dtypes(include=[np.number]).columns

        # Filter the columns to only scale those specified by the user
        columns_to_scale = [col for col in column_names if col in numeric_columns]

        scaler = StandardScaler()
        
        # Scale the specified columns
        scaled_data[columns_to_scale] = scaler.fit_transform(scaled_data[columns_to_scale])

        if show:
            print(scaled_data.head(15))

        self.df = scaled_data  # Update the original DataFrame with scaled values
        
        print(f'--- feature_scaler() complete (DataFrame input)\n')
        return self

    def encode_features(
    self,
    model_type: str,
    ordinal_cols: Union[list, dict] = None,
    categorical_cols: list = None,
    auto_encode: bool = False,
    show: bool = False
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
        if model_type not in ['Regressions', 'Machine Learning']:
            raise ValueError(f"Unknown model_type: {model_type}")


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
        if show:
            print(self.df.head(15))
        return self, encoded_values_dict if encoded_values_dict['ordinal'] or encoded_values_dict['categorical'] else None

    def split(
    self,
    split_ratio: tuple = (),
    target_name: str = None,
    random_state: int = None,
    show: bool = False
    ) -> tuple:
        """
        Splits a DataFrame into training, validation, and optionally test sets based on the provided split ratios.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            split_ratio (tuple): If two values (train_ratio, validation_ratio), if three values (train_ratio, validation_ratio, test_ratio).
            target_name (str): The column name to be used as the target variable.
            random_state (int): Seed for reproducibility.

        Returns:
            tuple: 
                - For two ratios: (train_features, train_target, valid_features, valid_target)
                - For three ratios: (train_features, train_target, valid_features, valid_target, test_features, test_target)
        """
        if len(split_ratio) not in (1, 2, 3):
            raise ValueError("split_ratio must have 1, 2, or 3 elements.")

        # save relevant new inputs to self. 
        # string
        self.target_name = target_name
        self.target = self.df[target_name] if target_name else None
        # df
        self.features = self.df.drop(self.target_name, axis=1)

        if random_state is None:
            random_state = self.random_state
        else:
            pass

        print(f'Running data_splitter()...')
        print(f'df shape start: {self.df.shape}')
        if len(split_ratio) == 0 or split_ratio is None:
            print(f'(no splitting)')
            return self
        
        elif len(split_ratio) == 1:
            split_ratio = split_ratio[0]
            if split_ratio <= 1:
                self.df = self.downsample(n_rows=split_ratio*len(self.df))

                print(f'--- data_splitter() complete\n')
                if show:
                    print(self.df.head(15))
                return self
            
            elif split_ratio > 1:
                self.df = self.upsample(self.df, n_rows=split_ratio*len(self.df))

                print(f'--- data_splitter() complete\n')
                if show:
                    print(self.df.head(15))
                return self

        elif len(split_ratio) == 2:
            train_ratio, val_ratio = split_ratio

            # Split data into training and validation sets.
            df_train, df_valid = train_test_split(self.df, test_size=val_ratio, random_state=random_state)

            print(f"Shapes:\ndf_train: {df_train.shape}\ndf_valid: {df_valid.shape}")

            self.train_features = df_train.drop(self.target_name, axis=1)
            self.train_target = df_train[self.target_name]
            self.valid_features = df_valid.drop(self.target_name, axis=1)
            self.valid_target = df_valid[self.target_name]

            print(f'--- data_splitter() complete\n')
            if show:
                print(self.train_features.head(15))
                print(self.valid_features.head(15))
            return self

        elif len(split_ratio) == 3:
            train_ratio, val_ratio, test_ratio = split_ratio

            # First split: separate out the test set.
            df_temp, self.df_test = train_test_split(self.df, test_size=test_ratio, random_state=random_state)

            # Recalculate validation ratio relative to the remaining data (df_temp).
            new_val_ratio = val_ratio / (1 - test_ratio)

            self.df_train, self.df_valid = train_test_split(df_temp, test_size=new_val_ratio, random_state=random_state)

            print(f"new data shapes:\ndf_train: {self.df_train.shape}\ndf_valid: {self.df_valid.shape}\ndf_test: {self.df_test.shape}")

            self.train_features = self.df_train.drop(self.target_name, axis=1)
            self.train_target = self.df_train[self.target_name]
            self.valid_features = self.df_valid.drop(self.target_name, axis=1)
            self.valid_target = self.df_valid[self.target_name]
            self.test_features = self.df_test.drop(self.target_name, axis=1)
            self.test_target = self.df_test[self.target_name]

            print(f'--- data_splitter() complete\n')
            if show:
                print(self.train_features.head(15))
                print(self.valid_features.head(15))
                print(self.test_features.head(15))
            return self

        else:
            raise ValueError("split_ratio must be a tuple with 3 or fewer elements.")
        
    def get_split(
        self,
        which: str = "all",     # options: 'train', 'valid', 'test', 'all'
        columns: str = "both",      # options: 'features', 'target', 'both'
        show: bool = False
    ):
        """
        Returns the specified split portion(s) of the dataset.

        Parameters:
            which (str): Which split to return. Options are:
                - "train"
                - "valid"
                - "test"
                - "all" (default, returns everything)
            kind (str): What to return. Options are:
                - "features"
                - "target"
                - "both" (default)

        Returns:
            tuple or array:
                - If 'all' and 'both': returns tuple of all features and targets.
                - If a specific split and 'both': returns (features, target).
                - If kind is 'features' or 'target': returns just that.
        """

        if which == "train":
            if columns == "features":
                if show:
                    print(f'train_features shape: {self.train_features.head(15)}')
                return self.train_features
            elif columns == "target":
                if show:
                    print(f'train_target shape: {self.train_target.head(15)}')
                return self.train_target
            else:
                if show:
                    print(f'train_features shape: {self.train_features.head(15)}')
                    print(f'train_target shape: {self.train_target.head(15)}')
                return self.train_features, self.train_target

        elif which == "valid":
            if columns == "features":
                if show:
                    print(f'valid_features shape: {self.valid_features.head(15)}')
                return self.valid_features
            elif columns == "target":
                if show:
                    print(f'valid_target shape: {self.valid_target.head(15)}')
                return self.valid_target
            else:
                if show:
                    print(f'valid_features shape: {self.valid_features.head(15)}')
                    print(f'valid_target shape: {self.valid_target.head(15)}')
                return self.valid_features, self.valid_target

        elif which == "test":
            if not hasattr(self, "test_features") or self.test_features is None:
                raise ValueError("Test split not available. Did you use a 3-part split?")
            if columns == "features":
                if show:
                    print(f'test_features shape: {self.test_features.head(15)}')
                return self.test_features
            elif columns == "target":
                if show:
                    print(f'test_target shape: {self.test_target.head(15)}')
                return self.test_target
            else:
                if show:
                    print(f'test_features shape: {self.test_features.head(15)}')
                    print(f'test_target shape: {self.test_target.head(15)}')
                return self.test_features, self.test_target

        elif which == "all":
            if show:
                print(f'train_features shape: {self.train_features.head(15)}')
                print(f'train_target shape: {self.train_target.head(15)}')
                print(f'valid_features shape: {self.valid_features.head(15)}')
                print(f'valid_target shape: {self.valid_target.head(15)}')
                if hasattr(self, "test_features") and self.test_features is not None:
                    print(f'test_features shape: {self.test_features.head(15)}')
                    print(f'test_target shape: {self.test_target.head(15)}')
            return (
                self.train_features, self.train_target,
                self.valid_features, self.valid_target,
                self.test_features if hasattr(self, "test_features") else None,
                self.test_target if hasattr(self, "test_target") else None
            )

        else:
            raise ValueError(f"Unknown value for 'which': {which}")

    def vectorize(
            self,
            df=None,
            features=None,
            target=None,
            show=False):
        '''
        Vectorizes inputs and stores them in self.*
        '''
        

        if df is not None:
            self.df_vectorized = df.to_numpy() if isinstance(df, pd.DataFrame) else df
        
        if features is not None and target is not None:
            self.features_vectorized = features.to_numpy() if isinstance(features, pd.DataFrame) else features
            self.target_vectorized = target.to_numpy() if isinstance(target, (pd.Series, pd.DataFrame)) else target
            self.df_vectorized = df.to_numpy() if isinstance(np.concatenate((features, target.reshape(-1, 1)), axis=1), (pd.Series, pd.DataFrame)) else None
        
        if df is None and features is None and target is None:
            self.train_features_vectorized = self.train_features.to_numpy() if isinstance(self.train_features, (pd.Series, pd.DataFrame)) else self.train_features
            self.train_target_vectorized = self.train_target.to_numpy() if isinstance(self.train_target, (pd.Series, pd.DataFrame)) else self.train_target
            self.valid_features_vectorized = self.valid_features.to_numpy() if isinstance(self.valid_features, (pd.Series, pd.DataFrame)) else self.valid_features
            self.valid_target_vectorized = self.valid_target.to_numpy() if isinstance(self.valid_target, (pd.Series, pd.DataFrame)) else self.valid_target
            self.test_features_vectorized = self.test_features.to_numpy() if isinstance(self.test_features, (pd.Series, pd.DataFrame)) else self.test_features
            self.test_target_vectorized = self.test_target.to_numpy() if isinstance(self.test_target, (pd.Series, pd.DataFrame)) else self.test_target
            self.features_vectorized = self.features.to_numpy() if isinstance(self.features, (pd.Series, pd.DataFrame)) else self.features
            self.target_vectorized = self.target.to_numpy() if isinstance(self.target, (pd.Series, pd.DataFrame)) else self.target
            self.df_vectorized = np.concatenate((self.features_vectorized, self.target_vectorized.reshape(-1, 1)), axis=1)

        print(f'df: {self.df_vectorized.shape}, features: {self.features_vectorized.shape}, target: {self.target_vectorized.shape}')
        if show:
            print(f'--- vectorize() complete\n')
            if df is not None:
                print(f'df_vectorized shape: {self.df_vectorized[:15, :]}')
            elif features is not None and target is not None:
                print(f'features_vectorized shape: {self.features_vectorized[:15, :]}')
                print(f'target_vectorized shape: {self.target_vectorized[:15]}')

        return self


    def get_vectorized(
            self,
            which: str = "all",     # options: 'train', 'valid', 'test', 'all', 'df'
            columns: str = "both",      # options: 'features', 'target', 'both'
            show: bool = False
        ):
            """
            Returns the specified split portion(s) of the dataset.

            Parameters:
                which (str): Which split to return. Options are:
                    - "train"
                    - "valid"
                    - "test"
                    - "all" (default, returns everything)
                kind (str): What to return. Options are:
                    - "features"
                    - "target"
                    - "both" (default)

            Returns:
                tuple or array:
                    - If 'all' and 'both': returns tuple of all features and targets.
                    - If a specific split and 'both': returns (features, target).
                    - If kind is 'features' or 'target': returns just that.
            """

            if which == "train":
                if columns == "features":
                    if show:
                        print(f'train_features shape:\n{self.train_features_vectorized[:15, :]}')
                    return self.train_features_vectorized
                elif columns == "target":
                    if show:
                        print(f'train_target shape:\n{self.train_target_vectorized[:15]}')
                    return self.train_target_vectorized
                else:
                    if show:
                        print(f'train_features shape:\n{self.train_features_vectorized[:15, :]}')
                        print(f'train_target shape:\n{self.train_target_vectorized[:15]}')
                    return self.train_features_vectorized, self.train_target_vectorized

            elif which == "valid":
                if columns == "features":
                    if show:
                        print(f'valid_features shape:\n{self.valid_features_vectorized[:15, :]}')
                    return self.valid_features_vectorized
                elif columns == "target":
                    if show:
                        print(f'valid_target shape:\n{self.valid_target_vectorized[:15]}')
                    return self.valid_target_vectorized
                else:
                    if show:
                        print(f'valid_features shape:\n{self.valid_features_vectorized[:15, :]}')
                        print(f'valid_target shape:\n{self.valid_target_vectorized[:15]}')
                    return self.valid_features_vectorized, self.valid_target_vectorized

            elif which == "test":
                if not hasattr(self, "test_features") or self.test_features is None:
                    raise ValueError("Test split not available. Did you use a 3-part split?")
                if columns == "features":
                    if show:
                        print(f'test_features shape:\n{self.test_features_vectorized[:15, :]}')
                    return self.test_features_vectorized
                elif columns == "target":
                    if show:
                        print(f'test_target shape:\n{self.test_target_vectorized[:15]}')
                    return self.test_target_vectorized
                else:
                    if show:
                        print(f'test_features shape:\n{self.test_features_vectorized[:15, :]}')
                        print(f'test_target shape:\n{self.test_target_vectorized[:15]}')
                    return self.test_features_vectorized, self.test_target_vectorized

            elif which == "all":
                if show:
                    print(f'train_features shape:\n{self.train_features_vectorized[:15, :]}')
                    print(f'train_target shape:\n{self.train_target_vectorized[:15]}')
                    print(f'valid_features shape:\n{self.valid_features_vectorized[:15, :]}')
                    print(f'valid_target shape:\n{self.valid_target_vectorized[:15]}')
                    print(f'test_features shape:\n{self.test_features_vectorized[:15, :]}')
                    print(f'test_target shape:\n{self.test_target_vectorized[:15]}')
                return (
                    self.train_features_vectorized, self.train_target_vectorized,
                    self.valid_features_vectorized, self.valid_target_vectorized,
                    self.test_features_vectorized if hasattr(self, "test_features") else None,
                    self.test_target_vectorized if hasattr(self, "test_target") else None
                )
            
            elif which == "df":
                if columns == "features":
                    if show:
                        print(f'df_vectorized shape:\n{self.df_vectorized[:15, :-1]}')
                    return self.df_vectorized[:, :-1]
                elif columns == "target":
                    if show:
                        print(f'df_target_vectorized shape:\n{self.df_vectorized[:15, -1]}')
                    return self.df_vectorized[:, -1]
                elif columns == 'all' or columns is None:
                    if show:
                        print(f'df_vectorized shape:\n{self.df_vectorized[:15, :]}')
                    return self.df_vectorized
                elif columns == 'both':
                    if show:
                        print(f'df_vectorized features shape:\n{self.df_vectorized[:15, :-1]}')
                        print(f'df_vectorized target shape:\n{self.df_vectorized[:15, -1]}')
                    return self.df_vectorized[:, :-1], self.df_vectorized[:, -1]
                else:
                    raise ValueError(f"Unknown value for 'columns': {columns}")

            else:
                raise ValueError(f"Unknown value for 'which': {which}")







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

