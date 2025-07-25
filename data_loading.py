import pandas as pd

def load_data(url=None):
    if url is None:
        url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    return df