#view df
# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.width', 500)        # Increase horizontal width
# pd.set_option('display.max_colwidth', None) # Show full content of each column

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

def view(dfs, view=None):
    # Convert input to a dictionary of DataFrames if needed
    if isinstance(dfs, pd.DataFrame):
        print(f"columns={dfs.columns.tolist()}")
        dfs = {'df': dfs}  # Wrap single DataFrame in a dict with a default name
    elif isinstance(dfs, pd.Series):
        series_name = dfs.name if dfs.name is not None else 'Series'
        dfs = {series_name: dfs.to_frame()}
    else:
        print("Input must be a pandas DataFrame or Series.")
        return

    views = {
        "headers": [],
        "values": [],
        "missing values": [],
        "dtypes": [],
        "summaries": []
    }

    missing_cols = []

    for df_name, df in dfs.items():
        for col in df.columns:
            # col stats
            counts = df[col].value_counts()
            common_unique_values = counts.head(5).index.tolist() if not counts.empty else []
            unique_values = counts.index.tolist() if not counts.empty else []
            rare_count = counts.tail(5).iloc[-1] if not counts.empty else np.nan
            rare_unique_values = counts.tail(5).index.tolist() if not counts.empty else []
            minority_ratio = rare_count / counts.sum() if counts.sum() > 0 else np.nan
            series_count = df[col].count()
            no_values = len(df) - series_count
            total = no_values + series_count
            no_values_percent = (no_values / total) * 100 if total != 0 else 0           
            
            # Ensure we don't fail on empty columns
            if df[col].count() > 0:
                data_type = type(df[col].iloc[0])
            else:
                data_type = np.nan

            # views cols
            views["headers"].append({
                'DataFrame': f'{df_name}',
                'Column': col,
                'Common Values': common_unique_values,
                'Unique Values': unique_values
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
        'summaries': f"Drop duplicates? \nDataFrames: {list(dfs.keys())}\ndf.duplicated().sum() \ndf.drop_duplicates() \ndf.duplicated().sum() \n"
    }

    if view is None or view == "all":
        for view_name, view_data in views.items():
            print(f'{view_name}:\n{pd.DataFrame(view_data)}\n{code.get(view_name, "")}\n')
            
    elif view in views:
        print(f'{view}:\n{pd.DataFrame(views[view])}\n{code.get(view, "")}\n')
    else:
        print("Invalid view. Available views are: headers, values, dtypes, missing values, summaries, or all.")


def see(
    df: pd.DataFrame,
    cols: list = None,
    x: str = None,
    case: str = None,
    n: int = 10  # Number of top values to show for categorical/text data
):
    '''
    Visualize a DataFrame:
    - Categorical: Bar Chart of top n values
    - Ordinal: Bar Chart of top n values
    - Continuous: Histogram with theoretical normal distribution on second axis
    - Text: Bar Chart of top n values
    '''
    # Determine the x-axis label based on the provided argument or index name
    if x is None:
        try:
            x_label = df.index.name or "Index"
        except:
            x_label = "Index"
    else:
        try:
            x_label = x.name or "Index"
        except AttributeError:
            x_label = "Index"

    # Ensure cols is a list
    if cols is None:
        cols = df.columns
    elif isinstance(cols, str):
        cols = [cols]
    
    # Create a color map for different lines
    color_map = plt.cm.get_cmap('tab10', len(cols))

    if case is None:
        # Loop through each column and create a separate plot
        for i, col in enumerate(cols):
            fig, ax = plt.subplots(figsize=(12, 6))
            dtype = df[col].dtype

            if pd.api.types.is_numeric_dtype(dtype) and df[col].nunique() >= 10:
                # Continuous data: Histogram with normal distribution
                sns.histplot(df[col], bins=30, kde=False, color=color_map(i), ax=ax)
                mu, std = norm.fit(df[col].dropna())
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                ax2 = ax.twinx()
                ax2.plot(x, p * (df[col].dropna().max() - df[col].dropna().min()), color=color_map(i+1))
                ax2.set_ylabel(f'Normal dist fit: $\mu$={mu:.2f}, $\sigma$={std:.2f}', color=color_map(i+1))
                ax.set_title(f'Histogram and Normal Distribution Fit of {col} by {x_label}')
                ax.set_xlabel(x_label)
                ax.set_ylabel(col, color=color_map(i))
            elif pd.api.types.is_numeric_dtype(dtype):
                # Continuous data but not enough unique values for normal distribution
                sns.histplot(df[col], bins=30, kde=False, color=color_map(i), ax=ax)
                ax.set_title(f'Histogram of {col} by {x_label}')
                ax.set_xlabel(x_label)
                ax.set_ylabel(col, color=color_map(i))
            else:
                # Categorical/Ordinal/Text data: Bar Chart of top n values
                value_counts = df[col].value_counts().head(n)
                value_counts.plot(kind='bar', color=color_map(i), ax=ax)
                ax.set_title(f'Top {n} Values of {col} by {x_label}')
                ax.set_xlabel(x_label)
                ax.set_ylabel('Count', color=color_map(i))
            
            ax.tick_params(axis='y', labelcolor=color_map(i))
            ax.grid(True)
            plt.legend(loc='best')
            plt.show()

# Example usage
# df = pd.DataFrame({
#     'A': np.random.normal(0, 1, 100),
#     'B': np.random.randint(0, 10, 100),
#     'C': ['apple', 'banana', 'apple', 'cherry', 'banana'] * 20
# })
# see(df, cols=['A', 'B', 'C'], x='Index')


    # elif case == 'graph_scores':
        # model_name, accuracy, precision, recall, f1 = model_scores
        # plt.figure(figsize=(8, 6))
        # plt.plot(model_scores['Recall'], model_scores['Precision'], label="Precision-Recall Curve")
        # plt.scatter(best_recall, best_precision,
        #             color='red',
        #             label=f'Max F1 = {max_f1:.2f}\nThreshold = {best_threshold:.2f}',
        #             zorder=5)
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.title("Precision-Recall Curve with Max F1 Indication")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
    else:
        pass
