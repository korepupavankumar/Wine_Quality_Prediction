from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_target(df):
    df["quality_label"] = df["quality"].apply(
        lambda x: 1 if x >= 7 else 0
    )
    return df


def split_data(df):

    X = df.drop(["quality", "quality_label"], axis=1)

    y = df["quality_label"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def scale_data(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled