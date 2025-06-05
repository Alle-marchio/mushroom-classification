import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Label encoding per tutte le colonne
    le = LabelEncoder()
    for column in df.columns:
        df[column] = le.fit_transform(df[column])

    X = df.drop('class', axis=1)
    y = df['class']

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scaling (solo per KNN, SVM, etc.)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
