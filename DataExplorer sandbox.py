'''
get DataExplorer() to work
'''
import pandas as pd
from DataExplorer import DataExplorer

df = pd.read_csv('data/sprint 8 churn.csv')


# test 1:
df = DataExplorer(df)

# return first column view and one column
df.view(view='headers', column=3).see()

# return first column view and one column
# df.view(view='headers', return_column=0)


# # return first column view and chart
# df.view(view='values').see()[0]

# # return first column view and chart
# df.view(view='missing_values').see()[0]

# # return first column view and chart
# df.view(view='dtypes').see()[0]

# # return first column view and chart
# df.view(view='summaries').see()[0]

# # test 2: loop through columns
# df.view(view='headers').see()[1]

# # test 3: return all columns
# df.view(view='headers').see()

# # test 4: return just tables
# df.view(view='headers')

# # test 5: return just charts
# df.see()[0]

# # test 6: return all column charts
# df.see(chart='bar')