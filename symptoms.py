# symptoms.py
DISEASE_SYMPTOMS = {
    "Heart Disease": [
        "Chest pain or pressure",
        "Shortness of breath",
        "Pain in left arm or shoulder",
        "Fatigue or unusual weakness",
        "Fast or irregular heartbeat",
        "Dizziness or lightheadedness",
        "Nausea or vomiting",
        "Cold sweats",
    ],
    "Diabetes": [
        "Frequent urination",
        "Excessive thirst",
        "Blurred vision",
        "Slow healing wounds or cuts",
        "Tingling or numbness in hands/feet",
        "Unexplained weight loss",
        "Always feeling hungry",
        "Extreme fatigue",
    ],
    "Stroke": [
        "Sudden numbness in face, arm or leg",
        "Sudden confusion or trouble speaking",
        "Sudden vision problems in one or both eyes",
        "Sudden severe headache with no reason",
        "Dizziness or loss of balance",
        "Trouble walking or coordination problems",
    ],
    "High Blood Pressure": [
        "Frequent headaches",
        "Dizziness or vertigo",
        "Blurred or double vision",
        "Nosebleeds",
        "Shortness of breath",
        "Chest pain or tightness",
        "Pounding feeling in chest or neck",
    ],
    "High Cholesterol": [
        "Chest pain during physical activity",
        "Yellowish deposits near eyes or skin",
        "Leg pain or cramps while walking",
        "Xanthomas (fatty bumps under skin)",
        "No symptoms (silent condition)",
    ],
}


def match_symptoms(selected_symptoms):
    results = {}
    for disease, symptoms in DISEASE_SYMPTOMS.items():
        matched    = [s for s in selected_symptoms if s in symptoms]
        total      = len(symptoms)
        count      = len(matched)
        percentage = round((count / total) * 100, 1) if total > 0 else 0.0
        results[disease] = {
            "matched":    count,
            "total":      total,
            "percentage": percentage,
            "symptoms":   matched,
        }
    return dict(sorted(results.items(), key=lambda x: x[1]["percentage"], reverse=True))


def get_all_symptoms():
    all_symptoms = []
    for symptoms in DISEASE_SYMPTOMS.values():
        for s in symptoms:
            if s not in all_symptoms:
                all_symptoms.append(s)
    return sorted(all_symptoms)