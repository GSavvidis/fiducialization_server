import pandas as pd
from pretty_print import pretty_print


directory = "/home/gsavvidis/csv_files/"
df = pd.read_csv(directory + "matched_tj13s000_MPA_q29.csv", index_col=[0,1,2,3]) 

df = df.unstack(level=["subentry", "entry","DD_NPulses"])
df = df.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')

pretty_print(df)
