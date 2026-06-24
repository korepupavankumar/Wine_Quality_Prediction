import pandas as pd

def load_data():
    df = pd.read_csv("winequality (1).csv")
    return df