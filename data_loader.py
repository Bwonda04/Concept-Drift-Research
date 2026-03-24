import pandas as pd

def load_elec2(filepath='elec.csv'):
    # read the csv
    df = pd.read_csv(filepath)

    # separate features and label — order is preserved by default
    X = df.drop(columns=['class']).values  # numpy array, shape (45312, 6)
    y = df['class'].values                 # numpy array, shape (45312,)

    return X, y