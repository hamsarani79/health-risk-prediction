# model.py

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from preprocess import load_and_clean_data, FEATURES, TARGETS


def train_all_models(filepath="data/heart_disease_health_indicators_BRFSS2015.csv"):
    df = load_and_clean_data(filepath)

    for disease_name, target_col in TARGETS.items():
        print(f"\n--- Training model for: {disease_name} ---")

        available_features = [f for f in FEATURES if f != target_col]

        if target_col not in df.columns:
            print(f"Column '{target_col}' not found, skipping.")
            continue

        X = df[available_features]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred   = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {round(accuracy * 100, 2)}%")

        model_filename = f"model_{disease_name.lower().replace(' ', '_')}.pkl"
        joblib.dump((model, available_features), model_filename)
        print(f"Saved: {model_filename}")

    print("\n All models trained and saved!")


if __name__ == "__main__":
    train_all_models()
    