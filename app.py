import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open("Diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

MEDIANS = {
    "Glucose": 117.0,
    "BloodPressure": 72.0,
    "SkinThickness": 29.0,
    "Insulin": 125.0,
    "BMI": 32.3,
}

OUTLIER_LIMITS = {
    "Pregnancies": {'lower': -6.5, 'upper': 13.5},
    "SkinThickness": {'lower': 14.5, 'upper': 42.5},
    "Insulin": {'lower': 105.0, 'upper': 145.0},
    "DiabetesPedigreeFunction": {'lower': -0.3464, 'upper': 1.2306},
    "Age": {'lower': -1.5, 'upper': 66.5},
}

def preprocess_single_row(df):
    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
    for col in cols_with_zeros:
        val = df[col].iloc[0]
        if val == 0:
            df[col] = MEDIANS[col]

    for col, limits in OUTLIER_LIMITS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=limits['lower'], upper=limits['upper'])
        
    return df

def predict_outcome(Pregnancies, Glucose, BloodPressure, SkinThickness, 
                    Insulin, BMI, DiabetesPedigreeFunction, Age):
    
    input_df = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness, 
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]],
    columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    
    processed_df = preprocess_single_row(input_df)
    
    prediction = model.predict(processed_df)[0]
    
    if prediction == 1:
        return "Result: Diabetes"
    else:
        return "Result: No Diabetes"

inputs = [
    gr.Number(label="Pregnancies", info="Number of times pregnant (e.g., 0-17)"),
    gr.Number(label="Glucose", info="Oral glucose tolerance test (Normal: <140, Pre-diabetes: 140-199, Diabetes: >199)"),
    gr.Number(label="BloodPressure", info="Diastolic blood pressure (mm Hg) (Normal: <80, Hypertension 1: 80-89, Hypertension 2: >89)"),
    gr.Number(label="SkinThickness", info="Triceps skin fold thickness (mm) (Typical range: 10-50)"),
    gr.Number(label="Insulin", info="2-Hour serum insulin (mu U/ml) (Typical range: 15-276)"),
    gr.Number(label="BMI", info="Body Mass Index (Underweight: <18.5, Healthy: 18.5-24.9, Overweight: 25-29.9, Obese: >30)"),
    gr.Number(label="DiabetesPedigreeFunction", info="Diabetes pedigree function (Typical range: 0.08 - 2.42)"),
    gr.Number(label="Age", info="Age in years (e.g., 21-81)")
]

app = gr.Interface(
    fn=predict_outcome,
    inputs=inputs,
    outputs="text", 
    title="Pima Indians Diabetes Predictor",
    description="Enter patient details to assess diabetes risk using an XGBoost model."
)

app.launch(share=True)