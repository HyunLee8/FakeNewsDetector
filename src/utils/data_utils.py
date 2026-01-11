import pandas as pd
import os

def load_data():
    HERE = os.path.dirname(os.path.abspath(__file__))
    CSV_FAKE_PATH = os.path.join(HERE, 'Fake.csv')
    CSV_TRUE_PATH = os.path.join(HERE, 'True.csv')
    df_fake = pd.read_csv(CSV_FAKE_PATH)
    df_true = pd.read_csv(CSV_TRUE_PATH)

    df_true['status'] = 1
    df_fake['status'] = 0
    df = pd.concat([df_true, df_fake])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.drop(['text', 'subject', 'date'], axis=1, inplace=True)
    return df