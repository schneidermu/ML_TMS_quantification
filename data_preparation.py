import pandas as pd


def prepare_data(df: pd.DataFrame, num_tms: int = 3):
    X = df.loc[:, 50:]
    y = (df.n_tms.astype(int)).apply(lambda x: x if x <= num_tms else num_tms)
    return X, y
