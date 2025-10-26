import pandas as pd

def load_data(fake_path='Fake.csv', true_path='True.csv'):
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    df_true['status'] = 1
    df_fake['status'] = 0
    df = pd.concat([df_true, df_fake])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.drop(['text', 'subject', 'date'], axis=1, inplace=True)
    return df