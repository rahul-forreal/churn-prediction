import pandas as pd


DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def inspect_data(df):
    print("\n===== DATASET SHAPE =====")
    print(df.shape)

    print("\n===== COLUMNS =====")
    print(df.columns.tolist())

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DUPLICATES =====")
    print(df.duplicated().sum())

    print("\n===== TARGET DISTRIBUTION =====")
    print(df["Churn"].value_counts())

    print("\n===== TARGET PERCENTAGE =====")
    print(df["Churn"].value_counts(normalize=True) * 100)

    print("\n===== SAMPLE DATA =====")
    print(df.head())


if __name__ == "__main__":
    df = load_data()
    inspect_data(df)