'''
get DataExplorer() to work
'''
import pandas as pd
from DataExplorer import DataExplorer

df = pd.read_csv('data/sprint 8 churn.csv')


# test 1:
df = DataExplorer(df)

# return first column view and one column
df.view(view='headers', column=3).see() # ✅


# add tqms and time to train a model
# for start_date, end_date in tqdm(
#                 date_ranges,
#                 desc='Getting Data', 
#                 unit='symbol', 
#                 leave=False,
#                 colour='green',  
#                 ascii=(" ", "█"),
#                 ncols=100


# # test 5: return just charts
# df.see()[0]

# # test 6: return all column charts
# df.see(chart='bar')