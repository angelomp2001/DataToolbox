'''
Exploratory Data Anlaysis: view columns as a table of statistics, key values, or graphically
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class DataExplorer:
    def __init__(
            self,
            df: pd.DataFrame = None
        ): 
        if isinstance(df, pd.DataFrame):
            self.df = df
        else:
            raise ValueError("Input must be a pandas DataFrame.")

        # print(f'dtype: {type(df)}') # df
    
    def view(
            self,
            table: str = None,
            column: int = None,
            full_screen = True
        ):
        '''
        view tables of stats of columns
        '''
        # save to class
        self.table = table

        if full_screen:
            pd.set_option('display.max_columns', None)  # Show all columns
            pd.set_option('display.width', 500)        # Increase horizontal width
            pd.set_option('display.max_colwidth', None) # Show full content of each column
            pd.set_option('display.max_rows', None)        # Show all rows
        
        if table not in ["headers", "values", "missing_values", "dtypes", "summaries"]:
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

        for col in self.df.columns:
            # col stats
            counts = self.df[col].value_counts()
            common_unique_values = counts.head(5).index.tolist() if not counts.empty else []
            n_unique_values = self.df[col].nunique() if not counts.empty else 0
            rare_count = counts.tail(5).iloc[-1] if not counts.empty else np.nan
            rare_unique_values = counts.tail(5).index.tolist() if not counts.empty else []
            minority_ratio = rare_count / counts.sum() if counts.sum() > 0 else np.nan
            series_count = self.df[col].count()
            no_values = len(self.df) - series_count
            total = no_values + series_count
            no_values_percent = (no_values / total) * 100 if total != 0 else 0           
            
            # get column data type
            if self.df[col].count() > 0:
                data_type = type(self.df[col].iloc[0])
            else:
                data_type = np.nan

            # views cols
            views["headers"].append({
                'Column': col,
                'Common Values': common_unique_values,
                'Unique Values': n_unique_values
            })

            views["values"].append({
                'Column': col,
                'Rare Values': rare_unique_values,
                'Minority ratio': f'{minority_ratio:.02f}'
            })

            views["missing values"].append({
                'Column': col,
                'Series Count': series_count,
                'Missing Values (%)': f'{no_values} ({no_values_percent:.0f}%)'
            })

            views["dtypes"].append({
                'Column': col,
                'Common Values': common_unique_values,
                'Data Type': data_type,
            })

            views["summaries"].append({
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
        if table is None or table == "all":
                for view_name, view_data in views.items():
                    # Create DataFrame and set index
                    df_view = pd.DataFrame(view_data)
                    df_view.index = range(len(views[table]))
                    self.df_view = df_view
                    if column is not None:
                        self.column = column
                        print(f'{view_name}:\n{df_view.iloc[column]}\n{code.get(view_name, "")}\n')  
                    else:
                        print(f'{view_name}:\n{df_view}\n{code.get(view_name, "")}\n')
                return self

        elif table in views:
            # Create specific view DataFrame and set index
            df_view = pd.DataFrame(views[table])
            df_view.index = range(len(views[table]))
            self.df_view = df_view
            if column is not None:
                self.column = column
                print(f'{table}:\n{df_view.iloc[column]}\n{code.get(table, "")}\n')
            else:
                print(f'{table}:\n{df_view}\n{code.get(table, "")}\n')
            return self  
        
        else:
            print("Invalid view. Available views are: headers, values, dtypes, missing values, summaries, or all.")

    def see(
        self,
        column: int = None,
        x_axis_name: str = None,
        n: int = 10  # Number of top values to show for categorical/text data
        ):
        '''
        Visualize a DataFrame:
        - Categorical: Bar Chart of top n values
        - Ordinal: Bar Chart of top n values
        - Continuous: Histogram with theoretical normal distribution on second axis
        - Text: Bar Chart of top n values
        
        column: target column to plot
        x: 
        '''
        ## validate inputs
        if column is None:
            if self.column is not None:
                cols = [self.column]
            else:
                cols = range(len(self.df.columns))
        elif isinstance(column, int):
            cols = [column]
        
        # Determine the x-axis label based on the provided argument or index name
        if x_axis_name is None:
            try:
                x_label = self.df[x_axis_name]
            except:
                x_label = "Index"
        else:
            try:
                x_label = x_axis_name
            except AttributeError:
                x_label = "Index"
        
        ## Color map for different lines
        color_map = plt.cm.get_cmap('tab10', len(self.df.iloc[:,max(cols)]))

        # Plot each column
        for i, col in enumerate(cols):

            # layout
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # plot by col dtype
            dtype = self.df.iloc[:,col].dtype
            # if col is continuous, plot histogram with normal distribution
            
            if x_axis_name is None: 
                
                if pd.api.types.is_numeric_dtype(dtype) and self.df.iloc[:,col].nunique() >= 10:
                    sns.histplot(self.df.iloc[:,col], bins=30, kde=False, color=color_map(i), ax=ax)
                    
                    # Normal distribution parameters
                    mu, std = norm.fit(self.df.iloc[:,col].dropna())

                    # get plot x min and max
                    xmin, xmax = plt.xlim()
                    
                    # calculates 100 evenly spaced values for PDF calculation
                    x = np.linspace(xmin, xmax, 100)
                    
                    # calculate PDF values
                    p = norm.pdf(x, mu, std)

                    # create second y axis on same x axis
                    ax2 = ax.twinx()

                    # plot normalized PDF (subtract min from all values, then multiple by p)
                    ax2.plot(x, p * (self.df.iloc[:,col].dropna().max() - self.df.iloc[:,col].dropna().min()), color=color_map(i+1))

                    # y axis label
                    ax2.set_ylabel(f'Normal dist fit: $\mu$={mu:.2f}, $\sigma$={std:.2f}', color=color_map(i+1))

                    # set title and labels
                    ax.set_title(f'Histogram and Normal Distribution Fit of {self.df.columns[col]} by {x_label}')
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(self.df.columns[col], color=color_map(i))
                
                # if col is continuous, plot histogram
                elif pd.api.types.is_numeric_dtype(dtype):
                    # Continuous data but not enough unique values for normal distribution
                    sns.histplot(self.df.iloc[:,col], bins=30, kde=False, color=color_map(i), ax=ax)

                    # set title and labels
                    ax.set_title(f'Histogram of {col} by {x_label}')
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(self.df.columns[col], color=color_map(i))
                else:
                    # plot Categorical/Ordinal/Text data: Bar Chart of top n values
                    # calculate value counts
                    value_counts = self.df.iloc[:,col].value_counts().head(n)
                    value_counts.plot(kind='bar', color=color_map(i), ax=ax)

                    # set title and labels
                    ax.set_title(f'Top {n} Values of {self.df.columns[col]} by {x_label}')
                    ax.set_xlabel(x_label)
                    ax.set_ylabel('Count', color=color_map(i))

            else:
                # if x_axis_name is provided
                # Set up the color map
                color_map = plt.cm.get_cmap('tab10', 2)  # Just two for the x and y axes

                # Layout
                fig, ax = plt.subplots(figsize=(12, 6))

                # Plot as a scatter plot
                ax.scatter(self.df[x_label], self.df.iloc[:,col], color=color_map(0), alpha=0.6)

                # Set title and labels
                ax.set_title(f'Scatter Plot of {col} vs {x_label}')
                ax.set_xlabel(x_label)
                ax.set_ylabel(self.df.columns[col], color=color_map(0))

            # set chart params (ticks, grid, legend)
            ax.tick_params(axis='y', labelcolor=color_map(i))
            ax.grid(True)
        
        plt.legend(loc='best')
        plt.show()
