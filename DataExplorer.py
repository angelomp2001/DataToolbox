'''
Exploratory Data Anlaysis: view columns as a table of statistics or graphically
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class DataExplorer:
    def __init__(
            self,
            df: pd.DataFrame or dict = None
    ): 
        if isinstance(df, dict):
            for _, each_df in df.items():
                if isinstance(each_df, pd.DataFrame):
                    self.df = {key: value  for key, value in df.items()}
                else:
                    raise ValueError("Input must be a pandas DataFrame.")
        elif isinstance(df, pd.DataFrame):
            self.df = {'df': df}
        else:
            raise ValueError("Input must be a pandas DataFrame.")

        # print(f'dtype: {type(df)}') # df
    
    def view(
            self,
            view: str = None,
            return_column: int = None,
            full_screen = True
        ):
        '''
        view tables of stats of columns
        '''
        # save to class
        self.view = view


        if full_screen:
            pd.set_option('display.max_columns', None)  # Show all columns
            pd.set_option('display.width', 500)        # Increase horizontal width
            pd.set_option('display.max_colwidth', None) # Show full content of each column
            pd.set_option('display.max_rows', None)        # Show all rows

        if view not in ["headers", "values", "missing_values", "dtypes", "summaries"]:
            raise ValueError("Invalid view. Available views are: headers, values, missing_values, dtypes, summaries, or all.")

        views = {
            "headers": [],
            "values": [],
            "missing values": [],
            "dtypes": [],
            "summaries": []
        }

        # initialize var holding missing % per column
        missing_cols = []


        for df_name, df in self.df.items():

            for col in df.columns:
                # col stats
                counts = df[col].value_counts()
                common_unique_values = counts.head(5).index.tolist() if not counts.empty else []
                n_unique_values = df[col].nunique() if not counts.empty else 0
                rare_count = counts.tail(5).iloc[-1] if not counts.empty else np.nan
                rare_unique_values = counts.tail(5).index.tolist() if not counts.empty else []
                minority_ratio = rare_count / counts.sum() if counts.sum() > 0 else np.nan
                series_count = df[col].count()
                no_values = len(df) - series_count
                total = no_values + series_count
                no_values_percent = (no_values / total) * 100 if total != 0 else 0           
                
                # get column data type
                if df[col].count() > 0:
                    data_type = type(df[col].iloc[0])
                else:
                    data_type = np.nan

                # views cols
                views["headers"].append({
                    'DataFrame': f'{df_name}',
                    'Column': col,
                    'Common Values': common_unique_values,
                    'Unique Values': n_unique_values
                })

                views["values"].append({
                    'DataFrame': f'{df_name}',
                    'Column': col,
                    'Rare Values': rare_unique_values,
                    'Minority ratio': f'{minority_ratio:.02f}'
                })

                views["missing values"].append({
                    'DataFrame': f'{df_name}',
                    'Column': col,
                    'Series Count': series_count,
                    'Missing Values (%)': f'{no_values} ({no_values_percent:.0f}%)'
                })

                views["dtypes"].append({
                    'DataFrame': f'{df_name}',
                    'Column': col,
                    'Common Values': common_unique_values,
                    'Data Type': data_type,
                })

                views["summaries"].append({
                    'DataFrame': f'{df_name}',
                    'Column': col,
                    'Common Values': common_unique_values,
                    'Rare Values': rare_unique_values,
                    'Data Type': data_type,
                    'Series Count': series_count,
                    'Missing Values': f'{no_values} ({no_values_percent:.0f}%)'
                })

                if no_values > 0:
                    missing_cols.append(col)

        code = {
            'headers': "Drop/rename?\ndf.rename(columns={col: col.lower() for col in df.columns}, inplace=True)",
            'values': "Manually fix missing? encode ordered/categorical?\n df['column_name'].replace(to_replace='old_value', value=None, inplace=True)\n# df['col_1'] = df['col_1'].fillna('Unknown', inplace=False)",
            'missing values': f"lots of missing?\n# Check for duplicates or summary statistics\nMissing Columns: {missing_cols}",
            'dtypes': "change dtype?\n# df['col'] = df['col'].astype(str) (Int64), (float64) \n# df['col'] = pd.to_datetime(df['col'], format='%Y-%m-%dT%H:%M:%SZ')",
            'summaries': f"Drop duplicates? \nDataFrames: {list(self.df.keys())}\ndf.duplicated().sum() \ndf.drop_duplicates() \ndf.duplicated().sum() \n"
        }

        # output logic
        if view is None or view == "all":
                for view_name, view_data in views.items():
                    # Create DataFrame and set index
                    df_view = pd.DataFrame(view_data)
                    df_view.index = range(len(views[view]))
                    self.df_view = df_view
                    if return_column is not None:
                        print(f'{view_name}:\n{df_view.iloc[return_column]}\n{code.get(view_name, "")}\n')  
                    else:
                        print(f'{view_name}:\n{df_view}\n{code.get(view_name, "")}\n')
                return self

        elif view in views:
            # Create specific view DataFrame and set index
            df_view = pd.DataFrame(views[view])
            df_view.index = range(len(views[view]))
            self.df_view = df_view
            if return_column is not None:
                print(f'{view}:\n{df_view.iloc[return_column]}\n{code.get(view, "")}\n')
            else:
                print(f'{view}:\n{df_view}\n{code.get(view, "")}\n')
            return self  
        
        else:
            print("Invalid view. Available views are: headers, values, dtypes, missing values, summaries, or all.")