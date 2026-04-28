import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_model(df, target, sensitive):
    # Keep sensitive separately
    sensitive_data = df[sensitive]

    # Prepare features
    X = df.drop(columns=[target, sensitive])
    y = df[target]

    # Convert categorical → numeric
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive_data, test_size=0.3, random_state=42
    )

    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    result_df = X_test.copy()
    result_df[target] = y_test.values
    result_df["prediction"] = preds
    result_df[sensitive] = s_test.values

    return result_df