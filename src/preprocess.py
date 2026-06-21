import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(df, test_size=0.2, random_state=42):
    # 1. Make a copy to avoid modifying the original
    data = df.copy()

    # 2. Drop the customer ID
    if 'customerID' in data.columns:
        data = data.drop('customerID', axis=1)

    # 3. Handle TotalCharges: convert to numeric, coerce errors, fill NaN with 0
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(0)  # or .mean()

    # 4. Encode target: 'Yes' -> 1, 'No' -> 0
    y = data['Churn'].map({'Yes': 1, 'No': 0})
    X = data.drop('Churn', axis=1)

    # 5. One‑hot encode categorical features
    #    (automatically converts object/string columns to dummy variables)
    X = pd.get_dummies(X, drop_first=True)  # drop_first avoids multicollinearity
    scaler = StandardScaler()

    X = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    # 6. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test