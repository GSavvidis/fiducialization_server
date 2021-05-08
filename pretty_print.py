
import pandas as pd

def pretty_print(df, max_rows = 100, max_cols = None, head=15 ):
    
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_cols):  # more options can be specified also
                print(df.head(head))

    return df
