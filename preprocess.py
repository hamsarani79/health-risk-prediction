# preprocess.py

import pandas as pd

def load_and_clean_data(filepath):
    filepath='data/heart_disease_health_indicators_BRFSS2015.csv'
    df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Smoker":            "Smoking",
        "PhysActivity":      "PhysicalActivity",
        "HvyAlcoholConsump": "AlcoholDrinking",
        "DiffWalk":          "DiffWalking",
        "Age":               "AgeCategory",
        "MentHlth":          "MentalHealth",
        "PhysHlth":          "PhysicalHealth",
    })

    df = df.dropna()
    print("Dataset loaded! Shape:", df.shape)
    return df

FEATURES = [
    "BMI", "Smoking", "AlcoholDrinking", "PhysicalActivity",
    "Stroke", "DiffWalking", "Sex", "AgeCategory", "Education",
    "HighBP", "HighChol", "Fruits", "Veggies"
]

TARGETS = {
    "Heart Disease":      "HeartDiseaseorAttack",
    "Diabetes":           "Diabetes",
    "Stroke":             "Stroke",
    "High Blood Pressure":"HighBP",
    "High Cholesterol":   "HighChol",
}