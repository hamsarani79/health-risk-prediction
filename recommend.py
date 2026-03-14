# recommend.py

DISEASE_TIPS = {
    "Heart Disease": [
        " Quit smoking — risk drops 50% within 1 year",
        " Exercise at least 30 minutes daily",
        " Eat more fruits, vegetables and whole grains",
        " Monitor blood pressure and cholesterol regularly",
        " Get 7-8 hours of sleep every night",
    ],
    "Diabetes": [
        " Reduce sugar and refined carbohydrates",
        " Maintain a healthy BMI",
        " Physical activity helps control blood sugar",
        " Get HbA1c tested every 3 months",
        " Stay well hydrated throughout the day",
    ],
    "Stroke": [
        " Control your blood pressure regularly",
        " Stop smoking immediately",
        " Avoid heavy alcohol consumption",
        " Stay physically active daily",
        " Know FAST signs: Face drooping, Arm weakness, Speech difficulty, Time to call",
    ],
    "High Blood Pressure": [
        " Reduce salt intake in your diet",
        " Exercise regularly to lower BP naturally",
        " Practice stress relief like deep breathing or yoga",
        " Take BP medications as prescribed",
        " Check blood pressure at least once a month",
    ],
    "High Cholesterol": [
        " Eat oats, nuts and olive oil — they reduce bad cholesterol",
        " Avoid fried and processed foods",
        " Regular exercise raises good HDL cholesterol",
        " Get a lipid panel blood test yearly",
        " Include omega-3 rich foods like fish in your diet",
    ],
}


def get_tips(disease_name):
    return DISEASE_TIPS.get(disease_name, [" Consult a doctor for personalized advice."])


def get_lifestyle_tips(user_data):
    tips = []
    if user_data.get("Smoking") == 1:
        tips.append(" You smoke — quitting is the single best thing you can do for your health.")
    if user_data.get("AlcoholDrinking") == 1:
        tips.append(" Reduce alcohol intake — it affects heart, liver and BP.")
    if user_data.get("PhysicalActivity") == 0:
        tips.append(" Start with 30 min walks daily — it reduces risk of all 5 diseases.")
    if user_data.get("BMI", 0) >= 30:
        tips.append(f" BMI {user_data['BMI']:.1f} is in obese range — weight loss helps greatly.")
    elif user_data.get("BMI", 0) >= 25:
        tips.append(f" BMI {user_data['BMI']:.1f} is overweight — a balanced diet will help.")
    if user_data.get("Fruits") == 0:
        tips.append(" Eat at least 1 serving of fruit daily for heart health.")
    if user_data.get("Veggies") == 0:
        tips.append(" Include vegetables in every meal — they reduce cholesterol and BP.")
    if not tips:
        tips.append(" Great lifestyle! Keep up regular check-ups and healthy habits.")
    return tips